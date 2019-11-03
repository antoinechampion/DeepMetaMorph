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

data_path = "../extracted/"
json <- NULL
df <- NULL
all_sequences <- NULL
all_instructions <- NULL

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
    json <- lapply(paste0(data_path, files), read_json)
    save_restore(option(json), "json")
  }
  # from json :
  # json[[file_name]][[seq_nb]][[inst_nb]]
  # [[ [[address, function, offset, hex_code, instruction_name, arguments]], 
  #    registers_before_instruction, registers_after_instruction, 
  #    stack_before_instruction, stack_after_instruction
  # ]]
  json <- sapply(json, function(x) ifelse(x == "NULL", NA, x))
  all_sequences <- unlist(json, recursive = FALSE)
  all_sequences[[1]] <- NULL
  # Remove empty sequences
  all_sequences <- list.clean(all_sequences, function(x) length(x) == 0)
  all_instructions <- unlist(all_sequences, recursive = FALSE)
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
  
  # Convert a list of 32-bits integers to a vector containing all their bits (little endian)
  parse_state <- function(regs, stack) {
    unlisted <- c(
      unlist(regs),
      unlist(stack)
    )
    m <- mapply(dectobin, unlisted)
    
    as.vector(m)
  }
  
  # Compute the difference between two states
  state_diff <- function(state_before, state_after) {
    state_after - state_before
  }
  
  # Replacement for standard strtoi/as.numeric cannot handle numbers greater than 2^31
  strtoi_ex <- function(x) {
    y <- str_replace(x, "0x", "")
    y <- as.numeric(paste0("0x", strsplit(y, "")[[1]]))
    sum(y * 16^rev((seq_along(y)-1)))
  }

  # Each row will be an instruction, the columns will be the following:
  # address, function, offset, hex_inst, instruction, arguments,
  # regs_before, regs_after, stack_before, stack_after
  print(strrep("-", 50))
  print("Processing your data...")
  print(" - Creating dataframe")
  df <- purrr::map_dfr(all_instructions, function(inst) {
    d <- data.frame(
        Address = inst[[1]][[1]],
        Function = inst[[1]][[2]],
        Offset = if (is.null(inst[[1]][[3]])) "0" else inst[[1]][[3]],
        HexCode = inst[[1]][[4]],
        Instruction = inst[[1]][[5]],
        Arguments = inst[[1]][[6]]
      )
    d$RegistersBefore <- list(lapply(unlist(inst[2]), strtoi_ex))
    d$RegistersAfter <- list(lapply(unlist(inst[3]), strtoi_ex))
    d$StackBefore <- list(lapply(unlist(inst[4]), strtoi_ex))
    d$StackAfter <- list(lapply(unlist(inst[5]), strtoi_ex))
    d
  })
  print(" - Parsing computer states")
  df$StateBefore <- mapply(parse_state, df$RegistersBefore,
                         df$StackBefore, SIMPLIFY = FALSE)
  df$StateAfter <- mapply(parse_state, df$RegistersAfter,
                           df$StackAfter, SIMPLIFY = FALSE)
  print(" - Computing state diffs")
  df$StateDiff <- mapply(state_diff, df$StateBefore, 
                         df$StateAfter, SIMPLIFY = FALSE)
  
  # Remove empty sequences
  m <- length(all_sequences)
  l <- max(sapply(all_sequences, length))
  n <- (length(df$StackBefore[[1]]) + length(df$RegistersBefore[[1]])) * 32
  seq_count <- length(all_sequences)
  seq_lengths <- lapply(all_sequences, length)
  inst_count <- length(unique(df, by = "Instruction"))
  inst_args_count <- length(all_instructions)
  
  # Drop useless columns
  df$RegistersBefore <- NULL
  df$RegistersAfter <- NULL
  df$StackBefore <- NULL
  df$StackAfter <- NULL
  
  print("All done.")
  print(strrep("-", 50))
}
stopifnot(dim(df)[[1]] == length(all_instructions))
# Add sequence number column
sequences_col <- c()
for (i in 1:length(all_sequences)) {
  sequences_col <- c(sequences_col, rep(i, length(all_sequences[[i]])))
}
df$Sequence <- sequences_col

theme_set(theme_bw())
data_analysis <- list()
data_analysis[["current_instruction"]] <- NA

# ------ ANALYSIS ------ 
# Instructions occurence count
data_analysis[["histogram"]] <- function() {
  ggplot(df, aes(x = Instruction)) +
    stat_count(width = 0.5) +
    geom_text(stat = 'count', aes(label = ..count..), vjust = -.5)
}

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
#
# Export mode 'seq2seq' exports every machine state from the start to the end of the sequence
# Export mode 'global' exports only the first and the last machine state of the sequence
# Export mode 'statediff' exports all the state diffs from the start to the end instead of machine states
data_analysis[["export"]] <- function(export_stack = TRUE, export_mode = "seq2seq") {
  print(strrep("-", 50))
  print("Freeing memory...")
  all_instructions <<- NA
  all_sequences <<- NA
  json <<- NA
  gc()
  
  print("Tidying input...")
  print("- Select")
  if (export_mode == "statediff") {
    X <- df %>% select(Sequence, StateDiff)
  } else {
    X <- df %>% select(Sequence, StateBefore, StateAfter)
  }
  df <<- NA
  gc()
  print("- Group By")
  X <- X %>% group_by(Sequence)
  print("- Mutate")
  X <- X %>% mutate(InstructionIndex = row_number())
  print("- Ungroup")
  X <- X %>% ungroup()
  print("- Pad sequences")
  
  if (export_mode == "seq2seq") {
    X_padded <- X %>% select(Sequence, InstructionIndex)
    X_padded$State <- X$StateBefore
    l <- l + 1
  } else if (export_mode == "statediff") {
    X_padded <- data.frame(X)
  }
  
  for (i in 1:m) {
    seq_l <- seq_lengths[[i]]
    if (seq_l < l) {
      to_pad <- l - seq_l
      pad <- data.frame(matrix(NA, nrow=to_pad, ncol=3))
      names(pad) <- names(X_padded)
      pad[["Sequence"]] <- rep(i,to_pad)
      if (export_mode == "statediff") {
        pad[["StateDiff"]] <- rep(list(rep(-1, n)), to_pad)
      }
      else if (export_mode == "seq2seq") {
        final_state <- X %>% filter(Sequence == i & InstructionIndex == seq_l)
        rep(list(rep(-1, n)), to_pad-1)
        pad[["State"]] <- c(list(final_state$StateAfter[[1]]),rep(list(rep(-1, n)), to_pad-1))
      }
      pad[["InstructionIndex"]] <- (seq_l+1):l
      X_padded <- rbind(X_padded, pad)
    }
  }
  print("- Arrange")
  X_padded <- X_padded %>% arrange(Sequence, InstructionIndex)
  return(X_padded)
  print("- Bind")
  if (export_mode == "statediff") {
    X_mat <- do.call(rbind, X_padded$StateDiff)
  } else {
    X_mat <- do.call(rbind, X_padded$State)
  }
  
  
  print("Exporting input...")
  write.csv(X_mat, file=gzfile("input.csv.gz", compression = 1), row.names = FALSE)
  
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
  print("- Pad sequences & add <GO> & <NOP> tokens")
  # Sequence are 2 step longer including an <GO> and an <NOP> token
  l <- l + 2
  Y_padded <- rep(NA, seq_count*l)
  for (i in 1:m) {
    if (i %% (m%/%100) == 0) {
      print(paste0(i %/% (m%/%100), "%"))
    }
    seq_l <- seq_lengths[[i]]
    to_pad <- l - seq_l - 1
    index_start <- ((i-1)*l+1)
    index_end <- (i*l)
    Y_padded[index_start] <- "<GO>"
    if (length(Y[Y$Sequence == i,]$InstAndArgs) == 0) {
      print(Y[Y$Sequence == i,]$InstAndArgs)
      print(i)
      print(seq_l)
    }
    Y_padded[(index_start+1):(index_start+seq_l)] <- Y[Y$Sequence == i,]$InstAndArgs
    Y_padded[(index_start+seq_l+1):index_end] <- rep("<NOP>", to_pad)
  }
  print("Exporting categorical output...")
  write.csv(Y_padded, file=gzfile("output.categorical.csv.gz", compression = 1), row.names = FALSE)
  
  # Creating dictionnary
  dict <- sort(unique(Y_padded))
  
  print("Exporting dictionary...")
  write.csv(dict, file=gzfile("dict.csv.gz", compression = 1), row.names = FALSE)
  
  print("- Exporting dimensions...")
  n_y <- length(dict)
  dims <- data.frame(m, l, l, n, n_y)
  colnames(dims) <- c("m", "T_x", "T_y", "n_x", "n_y")
  write.csv(dims, file=gzfile("dimensions.csv.gz", compression = 1), row.names = FALSE)
  
  print(">>>> All done. <<<<")
  print(strrep("-", 50))
}

data_analysis[["help"]] <- function() {
  print("You can use `data_analysis` object to see various statistics")
  print("  * `data_analysis[['histogram']]()` to view the histogram of instructions")
  print("  * `data_analysis[['instruction']](inst)` to start analysing the instruction 'inst' (NASM format)")
  print("  * `data_analysis[['heatmap_global']]()` to view the heatmap of the processor state for all the instructions")
  print("  * `data_analysis[['statistics']]()` to display various useful statistics")
  print("  * `data_analysis[['export']]()` to export the data for the model")
  print("  * `data_analysis[['help']]()` to display this message again. ")
}
data_analysis[["help"]]()