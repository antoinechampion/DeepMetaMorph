library(purrr)
library(magrittr)
library(optional)
library(dplyr)
library(reshape2)
library(ggplot2)
library(jsonlite)
library(binaryLogic)
library(stringr)
library(tidyr)
library(rlist)
library(redux)

# Parallel computing
library(parallel)
library(benchmarkme)
num_cores <- detectCores()
total_ram = benchmarkme::get_ram()
library(doParallel)
registerDoParallel(num_cores)
library(foreach)
library(future)
library(furrr)
future::plan(multiprocess)
print(paste0("Multicore enabled (", num_cores, " cores)"))
print("WARNING: This process is memory intensive, especially with multithreading enabled. You might want to run it in a control group to limit its memory usage") 

df <- NULL
all_sequences <- NULL
all_instructions <- NULL
data_analysis <- list()

# Load scrapped data from gdb
# data_path: folder containing gdb output
data_analysis[["load_gdb"]] <- function(data_path) {
  # ------ JSON PARSING ------ 
  {
    print(strrep("-", 50))
    print("Loading your data...")
    # Function to save and restore the variables already processed
    # if 'var' == none, then load the variable 'name' into the env
    # else, save 'var' to to the drive
    save_restore <- function(var, name) {
      path <- paste0(name, ".RData")
      
      if (!some(var)) {
        if (file.exists(path)) {
          print("Restoring existing data...")
          load(path, envir = environment())
          return(option(var))
        }
      }
      else {
        save(var, file = path)
      }
      
      return(none)
    }
    
    json <- save_restore(none, "json")
    if (!some(json)) {
      # If variables weren't saved, load data from extracted folder
      files <- list.files(path = data_path, pattern = "[0-9]+.*", recursive = TRUE)
      json <- mclapply(file.path(data_path, files), read_json, mc.cores=num_cores)
      save_restore(option(json), "json")
    }
    # JSON input format :
    # json[[file_name]][[seq_nb]][[inst_nb]]
    # [[ [[address, function, offset, hex_code, instruction_name, arguments]], 
    #    registers_before_instruction, registers_after_instruction, 
    #    stack_before_instruction, stack_after_instruction
    # ]]
    json <- sapply(json, function(x) ifelse(x == "NULL", NA, x))
    all_sequences <<- unlist(json, recursive = FALSE)
    all_sequences[[1]] <<- NULL
    # Remove empty sequences
    all_sequences <<- list.clean(all_sequences, function(x) length(x) == 0)
    all_instructions <<- unlist(all_sequences, recursive = FALSE)
    json <- NULL
    gc()
  }
  
  # ------ TIDY DATA ------ 
  {
    # 32 bits integer to a vector containing its binary representation
    dectobin <- function(x, size = 32) {
      if(x > 0) {
        c(dectobin(as.integer(x/2), size-1), x%%2)
      }
      else {
        integer(size)
      }
    }
    
    # Convert a list of 32-bits integers to their binary representation as vectors (little endian)
    parse_state <- function(regs, stack) {
      unlisted <- c(
        unlist(regs),
        unlist(stack)
      )
      # Too much overhead for multithreaded mapply here
      m <- mapply(dectobin, unlisted)
      
      as.vector(m)
    }
    
    # Compute the difference between two states
    state_diff <- function(state_before, state_after) {
      state_after - state_before
    }
    
    # Replacement for standard strtoi/as.numeric which cannot handle 
    # numbers greater than 2^31
    strtoi_ex <- function(x) {
      y <- str_replace(x, "0x", "")
      y <- as.numeric(paste0("0x", strsplit(y, "")[[1]]))
      sum(y * 16^rev((seq_along(y)-1)))
    }
    
    # Memory representation (dataframe) of the dataset:
    # Each row will be an instruction, the columns will be the following:
    # address, function, offset, hex_inst, instruction, arguments,
    # regs_before, regs_after, stack_before, stack_after
    print(strrep("-", 50))
    print("Processing your data...")
    print(" - Creating dataframe")
    df <<- future_map_dfr(all_instructions, function(inst) {
      d <- data.frame(
        Address = inst[[1]][[1]],
        Function = inst[[1]][[2]],
        Offset = if (is.null(inst[[1]][[3]])) "0" else inst[[1]][[3]],
        HexCode = inst[[1]][[4]],
        Instruction = inst[[1]][[5]],
        Arguments = inst[[1]][[6]]
      )
      # Too much overhead for multithreaded lapply here
      d$RegistersBefore <- list(lapply(unlist(inst[2]), strtoi_ex))
      d$RegistersAfter <- list(lapply(unlist(inst[3]), strtoi_ex))
      d$StackBefore <- list(lapply(unlist(inst[4]), strtoi_ex))
      d$StackAfter <- list(lapply(unlist(inst[5]), strtoi_ex))
      d
    })
    print(" - Parsing computer states")
    
    # RAM paging with threads in R is disastrous
    max_threads = max(1, total_ram %/% 16000000000)
    df$StateBefore <<- mcmapply(parse_state, df$RegistersBefore,
                                df$StackBefore, SIMPLIFY = FALSE,
                                mc.cores=max_threads)
    df$StateAfter <<- mcmapply(parse_state, df$RegistersAfter,
                               df$StackAfter, SIMPLIFY = FALSE,
                               mc.cores=max_threads)
    print(" - Computing state diffs")
    df$StateDiff <<- mcmapply(state_diff, df$StateBefore, 
                              df$StateAfter, SIMPLIFY = FALSE,
                              mc.cores=max_threads)
    
    # Remove empty sequences
    m <<- length(all_sequences)
    l <<- max(sapply(all_sequences, length))
    n_x <<- (length(df$StackBefore[[1]]) + length(df$RegistersBefore[[1]])) * 32
    seq_count <<- length(all_sequences)
    seq_lengths <<- mclapply(all_sequences, length, mc.cores=num_cores)
    inst_count <<- length(unique(df, by = "Instruction"))
    inst_args_count <<- length(all_instructions)
    
    # Drop useless columns
    df$RegistersBefore <<- NULL
    df$RegistersAfter <<- NULL
    df$StackBefore <<- NULL
    df$StackAfter <<- NULL
  }
  
  # Guard
  stopifnot(dim(df)[[1]] == length(all_instructions))
  
  # Add index column for sequence number
  sequences_col <- c()
  for (i in 1:length(all_sequences)) {
    sequences_col <- c(sequences_col, rep(i, length(all_sequences[[i]])))
  }
  df$Sequence <<- sequences_col
  
  theme_set(theme_bw())
  data_analysis[["current_instruction"]] <<- NA
  
  print("All done.")
  print(strrep("-", 50))
}

# ------ ANALYSIS ------ 
# Instructions occurrence count
data_analysis[["histogram"]] <- function() {
  ggplot(df, aes(x = Instruction)) +
    stat_count(width = 0.5) +
    geom_text(stat = 'count', aes(label = ..count..), vjust = -.5)
}

# Heat map of the bitwise usage of the program state for every instruction
plot_heatmap <- function(mat, plot_title) {
  melted <- melt(mat)
  p <- ggplot(melted, aes(x = Var2, y = Var1)) +
    geom_raster(aes(fill = value)) +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red") +
    labs(x = "", y = "",
         title = plot_title) +
    theme_bw() + xlim(0, 32) + ylim(0, 23)
  print(p)
}

data_analysis[["heatmap_instructions"]] <- function(i) {
  summarised <- df %>% group_by(HexCode) %>% summarise(mean = list(map(reduce(StateDiff, `map2`, `+`), `/`, n())), n = n())
  instruction <- summarised[summarised$HexCode == df$HexCode[[i]],]
  
  m <- matrix(unlist(instruction$mean[[1]]), nrow = 23, ncol = 32, byrow = TRUE)
  plot_heatmap(m, paste(df$Instruction[[i]], df$Arguments[[i]], "-", instruction$n[[1]], "occurence(s)"))
}

data_analysis[["heatmap_global"]] <- function() {
  m <- colMeans(abs(do.call(rbind, df$StateDiff)))
  dim(m) <- c(32, 23)
  plot_heatmap(t(m), paste("Machine state heatmap"))
}

data_analysis[["instruction"]] <- function(inst) {
  inst <- df[paste(df$Instruction, df$Arguments) == inst,]
  index <- as.numeric(rownames(inst))
  if (length(index > 1)) {
    data_analysis$heatmap_instructions(index[[1]])
  }
  else if (length(index) == 1) {
    data_analysis$heatmap_instructions(index)
  }
  else {
    print("This instruction doesn't exist")
    return()
  }
  data_analysis$current_instruction <<- inst
}

data_analysis[["statistics"]] <- function(inst) {
  print(strrep("-", 50))
  print(paste("Nb of sequences:", seq_count))
  print(paste("Nb of instructions:", inst_count))
  print(paste("Nb of (instruction, arguments) couples:", inst_args_count))
}

# ------ EXPORT ------ 
# Exports a 3-columns table (input), to input.csv
# - 1st column is the sequence index
# - 2nd column is the instruction index within the sequence
# - 3rd column are the parameters values (machine state before the instruction)
#
# Exports a 3-column table (output), to output.csv
# - 1st column is the sequence index
# - 2nd column is the instruction index within the sequence
# - 3rd column is the index (one hot) of the right instruction to use. 

data_analysis[["export_csv"]] <- function() {
  print(strrep("-", 50))
  print("Freeing memory...")
  all_instructions <<- NA
  all_sequences <<- NA
  json <<- NA
  gc()
  
  print("Tidying input...")
  print("- Select")
  X <- df %>% select(Sequence, StateBefore, StateAfter)
  print("- Group By")
  X <- X %>% group_by(Sequence)
  print("- Mutate")
  X <- X %>% mutate(InstructionIndex = row_number())
  print("- Ungroup")
  X <- X %>% ungroup()
  print("- Add final states")
  X$State <- df$StateBefore
  l <- l + 1
  l_sequence <- integer(length = m)
  l_instructionindex <- integer(length = m)
  l_state <- vector(mode = "list", length = m)
  foreach (i=1:m) %do% {
    l_sequence[[i]] <- i
    l_instructionindex[[i]] <- seq_lengths[[i]] + 1
    final_state <- X %>% filter(Sequence == i & InstructionIndex == seq_lengths[[i]])
    l_state[[i]] <- final_state$StateAfter[[1]]
  }
  print("- Bind")
  X <- X %>% bind_rows(
    tibble(Sequence = l_sequence, InstructionIndex = l_instructionindex, State = l_state)
  )
  print("- Select")
  X <- X %>% select(Sequence, InstructionIndex, State)
  print("- Arrange")
  X <- X %>% arrange(Sequence, InstructionIndex)
  print("- Format")
  X$State <- lapply(X$State, function(l){paste0(l, collapse="")})
  X <- X %>% group_by(Sequence) %>% 
    mutate(StateConcat = paste0(State, collapse=";")) %>% 
    filter(row_number()==1) %>%
    ungroup %>%
    select(StateConcat)
  print("Exporting input...")
  write.csv(X$StateConcat, file=gzfile("input.csv.gz", compression = 1), row.names = FALSE, quote=FALSE)
  
  X <<- NA
  gc()
  
  print("Tidying output...")
  inst_and_args <- unique(paste(df$Instruction, df$Arguments))
  
  print("- Mutate")
  Y <- df %>% mutate(InstAndArgs = paste(Instruction, Arguments))
  print("- Group By")
  Y <- Y %>% group_by(Sequence)
  print("- Mutate")
  Y <- Y %>% mutate(InstructionIndex = row_number())
  print("- Ungroup")
  Y <- Y %>% ungroup()
  print("- Select")
  Y <- Y %>% select(Sequence, InstructionIndex, InstAndArgs)
  print("- Add <GO> & <END> tokens")
  # Sequence are 2 step longer including an <GO> and an <NOP> token
  l <- l + 2
  l_sequence <- integer(length = m*2)
  l_instructionindex <- integer(length = m*2)
  l_instandargs <- vector(mode = "character", length = m*2)
  foreach (i=1:m) %do% {
    l_sequence[[i*2]] <- i
    l_instructionindex[[i*2]] <- 0
    l_instandargs[[i*2]] <- "<GO>"
    
    l_sequence[[i*2+1]] <- i
    l_instructionindex[[i*2+1]] <- seq_lengths[[i]] + 1
    l_instandargs[[i*2+1]] <- "<END>"
  }
  
  print("- Bind")
  Y <- Y %>% bind_rows(
    tibble(Sequence = l_sequence, InstructionIndex = l_instructionindex, InstAndArgs = l_instandargs)
  )
  Y <- Y %>% filter(Sequence != 0)
  print("- Select")
  Y <- Y %>% select(Sequence, InstructionIndex, InstAndArgs)
  print("- Arrange")
  Y <- Y %>% arrange(Sequence, InstructionIndex)
  print("- Format")
  Y_export <- Y %>% group_by(Sequence) %>% 
    mutate(InstConcat = paste0(InstAndArgs, collapse=";")) %>% 
    filter(row_number()==1) %>%
    ungroup %>%
    select(InstConcat)
  
  print("Exporting categorical output...")
  write.csv(Y_export, file=gzfile("output.categorical.csv.gz", compression = 1), row.names = FALSE, quote=FALSE)
  
  # Creating dictionnary
  dict <- sort(unique(Y$InstAndArgs))
  
  print("Exporting dictionary...")
  write.csv(dict, file=gzfile("dict.csv.gz", compression = 1), row.names = FALSE, quote=FALSE)
  
  print("- Exporting dimensions...")
  n_y <- length(dict)
  dims <- data.frame(m, l, l, n_x, n_y)
  colnames(dims) <- c("m", "T_x", "T_y", "n_x", "n_y")
  write.csv(dims, file=gzfile("dimensions.csv.gz", compression = 1), row.names = FALSE, quote=FALSE)
  
  print(">>>> All done. <<<<")
  print(strrep("-", 50))
}

# Send pre-processed data to redis. This function needs a listening redis server
# on localhost without authentication.
# use_csv: if you want to import existing already pre-processed data in 
#     .csv.gz format instead of exporting it again, set this argument to the 
#     folder path containing input.csv.gz, output.categorical.gz, 
#     dimensions.csv.gz and dict.csv.gz
data_analysis[["to_redis"]] <- function(use_csv = "") {
  redis_prefix <- "deepmm_"
  redis_data_name <- paste0(redis_prefix,"available_data")
  redis_dict_name <- paste0(redis_prefix,"dict")
  
  if (!redux::redis_available()) {
    print("Redis has not been found on the local machine.") 
    print("Aborting...")  
    return()
  }
  if (use_csv == "") {
    data_analysis[["export_csv"]]()
    use_csv <- "./"
  }
  
  print("Preparing data...")
  
  read_file_to_vector <- function(path) {
    f <- gzfile(path, open="rb")
    readLines(f, 1)
    data <- readLines(f)
    close(f)
    data
  }

  X <- read_file_to_vector(file.path(use_csv, "input.csv.gz"))
  Y <- read_file_to_vector(file.path(use_csv, "output.categorical.csv.gz"))
  dict <- read_file_to_vector(file.path(use_csv, "dict.csv.gz"))
  dimensions <- read.csv(gzfile(file.path(use_csv, "dimensions.csv.gz")))
  
  
  print("Exporting to redis...")
  r <- redux::hiredis()
  
  # Dataset
  # Fetch using:
  #   SPOP deepmm_available_data [batch size]
  #   SADD deepmm_used_data [...]
  XY <- paste(X, Y, sep="-")
  for (xy in XY) {
    r$SADD(redis_data_name, xy)
  }
  
  # Dictionary
  # Fetch using ZRANGEBYSCORE deepmm_dict -inf +inf
  for (i in 1:length(dict)) {
    r$ZADD(redis_dict_name, i, dict[[i]])
  }
  
  # Dimensions
  # Fetch using GET deepmm_[...]
  r$SET(paste0(redis_prefix,"m"), dimensions$m)
  r$SET(paste0(redis_prefix,"T_x"), dimensions$T_x)
  r$SET(paste0(redis_prefix,"T_y"), dimensions$T_y)
  r$SET(paste0(redis_prefix,"n_x"), dimensions$n_x)
  r$SET(paste0(redis_prefix,"n_y"), dimensions$n_y)
  print(">>>> All done. <<<<")
}

data_analysis[["help"]] <- function() {
  print("You can use `data_analysis` object to prepare data and to see various statistics")
  print("  * `data_analysis[['load_gdb']]()` to load scrapped data from gdb")
  print("  * `data_analysis[['histogram']]()` to view the histogram of instructions")
  print("  * `data_analysis[['instruction']](inst)` to start analysing the instruction 'inst' (NASM format)")
  print("  * `data_analysis[['heatmap_global']]()` to view the heatmap of the processor state for all the instructions")
  print("  * `data_analysis[['statistics']]()` to display various useful statistics")
  print("  * `data_analysis[['export_csv']]()` to export the data for the model")
  print("  * `data_analysis[['to_redis']](csv_path=NULL)` to export the data a local redis database")
  print("  * `data_analysis[['help']]()` to display this message again. ")
}
data_analysis[["help"]]()
