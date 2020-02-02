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
  # json[[seq_nb]][[inst_nb]]
  # [[ [[address, function, offset, hex_code, instruction_name, arguments]], 
  #    registers_before_instruction, registers_after_instruction, 
  #    stack_before_instruction, stack_after_instruction
  # ]]
  json <- sapply(json, function(x) ifelse(x == "NULL", NA, x))
  all_sequences <- unlist(json, recursive = FALSE)
  all_sequences[[1]] <- NULL
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
  
  # Convenience function to compute the bit by bit difference 
  # between two states of the processor
  state_diff <- function(regs_before, regs_after, stack_before, stack_after) {
    xb <- c(
    unlist(regs_after),
    unlist(stack_after)
    )
    yb <- c(
    unlist(regs_before),
    unlist(stack_before)
    )
    m <- mapply(function(k, l) {
      k <- lapply(k, dectobin)
      l <- lapply(l, dectobin)
      diff <- mapply(`-`, k, l)
      diff
    }, xb, yb)
    
    as.vector(m)
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
  print(" - Computing state diffs")
  df$StateDiff <- mapply(state_diff, df$RegistersBefore, df$RegistersAfter,
        df$StackBefore, df$StackAfter, SIMPLIFY = FALSE)
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
  plot_heatmap(t(m), paste("Global processor state heatmap"))
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
  print(paste("Nb of sequences:", length(all_sequences)))
  print(paste("Nb of instructions:", length(unique(df, by = "Instruction"))))
  print(paste("Nb of (instruction, arguments) couples:", length(all_instructions)))
}

# ------ EXPORT ------ 
# Exports a 3-columns table (input), to input.csv
# - 1st column is the sequence index
# - 2nd column is the instruction index within the sequence
# - 3rd column are the parameters values (state_diff of the instruction)
#
# Exports a 3-column table (output), to output.csv
# - 1st column is the sequence index
# - 2nd column is the instruction index within the sequence
# - 3rd column is the index (one hot) of the right instruction to use. 
data_analysis[["export"]] <- function(export_stack = TRUE) {
  m <- length(all_sequences)
  l <- max(sapply(all_sequences, length))
  n <- length(df$RegistersBefore[[1]]) * 32
  if (export_stack) {
    n <- n + length(df$StackBefore[[1]]) * 32
  }
  
  print(strrep("-", 50))
  print("Tidying input...")
  
  
  print("- Select")
  X <- df %>% select(Sequence, StateDiff)
  print("- Group By")
  X <- X %>% group_by(Sequence)
  print("- Mutate")
  X <- X %>% mutate(InstructionIndex = row_number())
  print("- Ungroup")
  X <- X %>% ungroup()
  print("- Pad sequences")
  X_padded <- data.frame(X)
  for (i in 1:m) {
    seq_l <- length(all_sequences[[i]])
    if (seq_l < l) {
      to_pad <- l - seq_l
      pad <- data.frame(matrix(NA, nrow=to_pad, ncol=3))
      names(pad) <- names(X)
      pad[["Sequence"]] <- rep(i,to_pad)
      pad[["StateDiff"]] <- rep(list(rep(-1, n)), to_pad)
      pad[["InstructionIndex"]] <- (seq_l+1):l
      X_padded <- rbind(X_padded, pad)
    }
  }
  print("- Arrange")
  X_padded <- X_padded %>% arrange(Sequence, InstructionIndex)
  print("- Bind")
  X_mat <- do.call(rbind, X_padded$StateDiff)
  
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
  print("- Pad sequences & add <GO> & <END> tokens")
  Y_padded <- data.frame(Y)
  for (i in 1:m) {
    seq_l <- length(all_sequences[[i]])
    if (seq_l < l) {
      to_pad <- l - seq_l
      pad <- data.frame(matrix(NA, nrow=to_pad, ncol=3))
      names(pad) <- names(Y)
      pad[["Sequence"]] <- rep(i,to_pad)
      pad[["InstructionIndex"]] <- (seq_l+1):l
      pad[["InstAndArgs"]] <- rep("<END>", to_pad)
      Y_padded <- rbind(Y_padded, pad)
      
      # 0 index will be the <GO> token as InstructionIndex starts
      # from 1
      go <- data.frame(matrix(NA, nrow=1, ncol=3))
      names(go) <- names(Y)
      go[["Sequence"]] <- i
      go[["InstructionIndex"]] <- 0
      go[["InstAndArgs"]] <- "<GO>"
      Y_padded <- rbind(Y_padded, go)
    }
  }
  # Sequence are 1 step longer including the <GO> token
  l <- l + 1
  print("- Arrange")
  Y_padded <- Y_padded %>% arrange(Sequence, InstructionIndex) %>% select(InstAndArgs)
  print("Exporting categorical output...")
  write.csv(Y_padded, file=gzfile("output.csv.gz", compression = 1), row.names = FALSE)
  
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
data_analysis[["export"]]()