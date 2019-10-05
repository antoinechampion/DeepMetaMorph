library(purrr)
library(magrittr)
library(optional)
library(dplyr)
library(reshape2)
library(ggplot2)
library(jsonlite)

data_path = "../extracted/"
json <- NULL
df <- NULL
all_sequences <- NULL
all_instructions <- NULL

# ------ JSON PARSING ------ 
{
  # Function to save and restore the variables already processed
  # if 'var' == none, then load the variable 'name' into the env
  # else, save 'var' to to the drive
  save_restore <- function(var, name) {
    path <- paste0(name, ".RData")

    if (!some(var)) {
      if (file.exists(path)) {
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
  # Convenience function to compute the bit by bit difference 
  # between two states of the processor
  state_diff <- function(regs_before, regs_after, stack_before, stack_after) {
    xb <- intToBits(c(
    unlist(regs_after),
    unlist(stack_after)
    ))
    yb <- intToBits(c(
    unlist(regs_before),
    unlist(stack_before)
    ))
    m <- mapply(function(k, l) {
      k = readBin(k, what = "int", size = 1)
      l = readBin(l, what = "int", size = 1)
      k - l
    }, xb, yb)
    list(m)
  }

  # Each row will be an instruction, the columns will be the following:
  # address, function, offset, hex_inst, instruction, arguments,
  # regs_before, regs_after, stack_before, stack_after
  dbg <- TRUE
  df <- purrr::map_dfr(all_instructions, function(inst) {
    d <- data.frame(
        Address = inst[[1]][[1]],
        Function = inst[[1]][[2]],
        Offset = if (is.null(inst[[1]][[3]])) "0" else inst[[1]][[3]],
        HexCode = inst[[1]][[4]],
        Instruction = inst[[1]][[5]],
        Arguments = inst[[1]][[6]]
      )
    d$RegistersBefore <- list(lapply(unlist(inst[2]), strtoi))
    d$RegistersAfter <- list(lapply(unlist(inst[3]), strtoi))
    d$StackBefore <- list(lapply(unlist(inst[4]), strtoi))
    d$StackAfter <- list(lapply(unlist(inst[5]), strtoi))
    d
  })
  df$StateDiff <- mapply(state_diff, df$RegistersBefore, df$RegistersAfter,
        df$StackBefore, df$StackAfter)
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
# Exports a m*l*n matrix (input), to deepmm-x.csv
# - m is the number of samples (nb of sequences)
# - l is the max number of instructions (in a sequence)
# - n is the number of parameters
# Parameters corresponds to all the bits of a state of the cpu
# before and after the instruction l of the sequence m
# Also exports a m*l matrix (output), to deepmm-y.csv
data_analysis[["export"]] <- function(as_state_diff = FALSE, export_stack = TRUE) {
  m <- length(all_sequences)
  l <- max(sapply(all_sequences, length))
  n <- (length(df$RegistersBefore[[1]]) + length(df$RegistersAfter[[1]])) * 32
  if (export_stack) {
    n <- n + (length(df$StackBefore[[1]]) + length(df$StackAfter[[1]])) * 32
  }
  print(strrep("-", 50))
  print(paste0("Exporting ", m, "*", n, "*", l, " matrix."))
  print(       "          ^--- number of sequences")
  print(       "                  ^--- number of instructions by sequence")
  print(       "                      ^--- number of parameters")
  if (as_state_diff) {
    return()
  }
  else {
    # PAS BON
    df$.ConcatParams <- within(df, id <- paste(df$RegistersBefore, df$StackBefore, df$RegisterAfter, df$StackAfter, sep = ""))
    # One line concatenated should have 7+7+16+16=46 parameters
    # ???
    dbg <- TRUE
    for (i in 1:length(df)) {
      bin_params <- list()
      for (j in 1:length(df$.ConcatParams[[i]])) {
        bits <- intToBits(unlist(df$.ConcatParams[[i]][[j]]))
        if (dbg) {
          print(df$.ConcatParams[[i]][[j]])
          print(bits)
          dbg <- FALSE
        }
        bin_params <- c(bin_params, bits)
      }
      df$.BinParams[[i]] <- bin_params
    }
    df$.BinParams <- sapply(df$.ConcatParams, function(x) {
      b <- sapply(x, function(y) {
        intToBits(unlist(y))
      })
      b <- sapply(b, function(y) {
        readBin(y, what = "int", size = 1)
      })
      list(b)
    })
  }
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