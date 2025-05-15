# ========= Libraries =========
library(purrr)
library(stringr)
library(reticulate)
library(pdftools)
library(dplyr)
library(stringr)
library(tidytext)
library(textclean)
library(textstem)
library(syuzhet)
library(topicmodels)
library(udpipe)
library(quanteda)
library(textrank)
library(ggplot2)
library(knitr)
library(kableExtra)
library(zoo)
library(forcats)

# ========= Setup =========
setwd("~/Desktop/Masters Thesis")
pdf_folder <- "~/Desktop/Masters Thesis/Dataset"

use_python("/Library/Frameworks/Python.framework/Versions/3.10/bin/python3", required = TRUE)
py_run_string("import fitz")

pdf_files <- list.files(pdf_folder, pattern = "\\.pdf$", full.names = TRUE)
py$pdf_files <- pdf_files

py_run_string("
import re
all_text = []
for path in pdf_files:
    text = ''
    with fitz.open(path) as doc:
        for page in doc:
            page_text = page.get_text()
            page_text = re.sub(r'\\x00', '', page_text)
            text += page_text
    all_text.append(text)
")

texts <- py$all_text
files <- basename(pdf_files)

# ========= Create Stage Data =========
text_df <- tibble(
  file = files,
  text = texts
) %>%
  mutate(year = as.numeric(str_extract(file, "\\d{4}"))) %>%
  mutate(stage = case_when(
    year == 2022 ~ "Stage 1: Disruption",
    year %in% c(2023, 2024) ~ "Stage 2: Destabilization",
    year == 2025 ~ "Stage 3: Reconstruction",
    TRUE ~ NA_character_
  )) %>%
  filter(!is.na(stage))

# ===== 2. Combine Text by Stage =====
stage_texts <- text_df %>%
  group_by(stage) %>%
  summarise(full_text = paste(text, collapse = " "), .groups = "drop")

# ===== 3. Define Cleaning Function for BERTopic Input =====
custom_clean <- function(text) {
  text %>%
    str_to_lower() %>%
    str_replace_all("[^[:alnum:]\\s]", " ") %>%
    str_replace_all("\\b(cookies?|website|privacy|terms|subscribe|click|read more|accept|policy|consent|our|you|we|the|and|to|of)\\b", "") %>%
    str_squish()
}

# ===== 4. Apply Cleaning to Each Stage =====
stage_texts <- stage_texts %>%
  mutate(cleaned_text = map_chr(full_text, custom_clean))

# ===== 5. Split Cleaned Text into 100-word Chunks and Save to TXT Files =====
for (i in 1:nrow(stage_texts)) {
  
  stage_name <- stage_texts$stage[i]
  clean_stage_name <- gsub(" ", "", stage_name)
  
  words <- unlist(str_split(stage_texts$cleaned_text[i], "\\s+"))
  chunks <- split(words, ceiling(seq_along(words) / 100))
  chunk_texts <- sapply(chunks, function(x) paste(x, collapse = " "))
  
  writeLines(chunk_texts, paste0("bertopic_", clean_stage_name, ".txt"))
  cat(paste("✅ Saved: bertopic_", clean_stage_name, ".txt\n", sep = ""))
}

text_df %>%
  count(stage, name = "num_articles")

lda_results <- list()

data("stop_words")

# DEFINE number words
number_words <- c("one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                  "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                  "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
                  "eighty", "ninety", "hundred", "thousand", "million", "billion")

# UDPipe Setup
model <- udpipe_download_model(language = "english")
ud_model <- udpipe_load_model(model$file_model)

text_df %>%
  count(stage, name = "number_of_articles")

# ========= LOOP through each Stage =========
for (i in 1:nrow(stage_texts)) {
  
  stage_name <- stage_texts$stage[i]
  stage_text <- stage_texts$full_text[i]
  
  cat("\n\nRunning analysis for:", stage_name, "\n")
  
  # --- Preprocessing ---
  clean_text <- stage_text %>%
    str_replace_all("[^[:alnum:]\\s]", " ") %>%
    str_squish() %>%
    lemmatize_strings()
  
  text_df <- tibble(text = clean_text)
  
  # --- Tokens ---
  tokens_df <- text_df %>%
    unnest_tokens(word, text) %>%
    filter(!word %in% stop_words$word,
           !str_detect(word, "^[0-9]+$"),
           !word %in% number_words,
           nchar(word) > 3)
  
  # --- Word Frequency ---
  top_words <- tokens_df %>%
    count(word, sort = TRUE) %>%
    slice_max(n, n = 40)
  
  print(
    ggplot(top_words, aes(x = reorder(word, n), y = n)) +
      geom_col(fill = "#607D8B") +
      coord_flip() +
      labs(title = paste("Top 40 Words -", stage_name), x = "Word", y = "Frequency")
  )
  
  # --- UDPipe POS Tagging ---
  anno <- udpipe_annotate(ud_model, x = clean_text)
  anno_df <- as.data.frame(anno)
  
  # --- Named Entity Recognition / Keyphrase Extraction ---
  ner_phrases <- textrank_keywords(
    anno_df$lemma,
    relevant = anno_df$upos %in% c("NOUN", "ADJ"),
    ngram_max = 3,
    sep = " "
  )
  
  # DEBUG: Check what's actually inside
  print("=== STRUCTURE OF ner_phrases$keywords ===")
  print(str(ner_phrases$keywords))
  
  # If keywords exist, process them
  if (!is.null(ner_phrases$keywords) && nrow(ner_phrases$keywords) > 0) {
    
    # Check column names just to be safe
    colnames(ner_phrases$keywords)
    
    # Use proper column name if not 'textrank'
    ner_top <- ner_phrases$keywords %>%
      rename(score = 2) %>%  # rename second column to generic 'score'
      arrange(desc(score)) %>%
      slice_head(n = 20)
    
    print(
      ggplot(ner_top, aes(x = reorder(keyword, score), y = score)) +
        geom_col(fill = "#90A4AE") +
        coord_flip() +
        labs(title = paste("Top Named Entities / Key Phrases -", stage_name),
             x = "Keyword / Entity", y = "Importance (TextRank)") +
        theme_minimal()
    )
    
  } else {
    warning(paste("⚠️ No key phrases extracted for", stage_name))
  }
  
  
  # --- Sentiment over Chunks ---
  words <- unlist(str_split(text_df$text, "\\s+"))
  chunks <- split(words, ceiling(seq_along(words) / 100))
  chunk_texts <- sapply(chunks, paste, collapse = " ")
  
  chunk_sentiment <- get_sentiment(chunk_texts)
  chunk_sentiment_smooth <- rollmean(chunk_sentiment, k = 50, fill = NA)
  
  # Plot Smoothed Sentiment
  plot(chunk_sentiment_smooth, type = "l",
       main = paste("Smoothed Sentiment -", stage_name),
       xlab = "Chunk", ylab = "Avg Sentiment")
  
  avg_sentiment <- mean(chunk_sentiment_smooth, na.rm = TRUE)
  max_idx <- which.max(chunk_sentiment_smooth)
  min_idx <- which.min(chunk_sentiment_smooth)
  
  abline(h = avg_sentiment, col = "darkblue", lty = 2)
  points(max_idx, chunk_sentiment_smooth[max_idx], col = "lightgreen", pch = 21, bg = "lightgreen", cex = 1.5)
  points(min_idx, chunk_sentiment_smooth[min_idx], col = "indianred", pch = 21, bg = "indianred", cex = 1.5)
  text(max_idx, chunk_sentiment_smooth[max_idx], labels = "Peak", pos = 3, col = "lightgreen", cex = 0.8)
  text(min_idx, chunk_sentiment_smooth[min_idx], labels = "Trough", pos = 3, col = "indianred", cex = 0.8)
  
  # --- NRC Sentiment Bar Chart ---
  sentiment_scores <- get_nrc_sentiment(stage_text)
  sentiment_totals <- colSums(sentiment_scores)
  sentiment_percent <- round(100 * sentiment_totals / sum(sentiment_totals), 1)
  
  # Define sentiment polarity groups
  positive_emotions <- c("positive", "joy", "trust", "anticipation")
  negative_emotions <- c("negative", "anger", "fear", "sadness", "disgust")
  
  # Assign bar colors based on polarity
  bar_colors <- ifelse(names(sentiment_totals) %in% positive_emotions, "#4CAF50",  # green for positive
                       ifelse(names(sentiment_totals) %in% negative_emotions, "#F44336",  # red for negative
                              "#B0BEC5"))  # neutral grey for other categories like surprise
  
  # Plot the bar chart
  bar_positions <- barplot(sentiment_totals,
                           las = 2,
                           main = paste("NRC Sentiment Distribution -", stage_name),
                           col = bar_colors,
                           ylim = c(0, max(sentiment_totals) * 1.2))
  
  # Add percentage labels above bars
  text(x = bar_positions,
       y = sentiment_totals,
       labels = paste0(sentiment_percent, "%"),
       pos = 3, cex = 0.8)
  
  
  # --- BERTopic Import (already run in Python/Colab) ---
  
  # Extract "Stage1", "Stage2", etc. from full stage name
  stage_label <- str_extract(stage_name, "Stage\\s*\\d")
  stage_label <- gsub(" ", "", stage_label)
  
  # Read the correct CSV file
  csv_path <- paste0("bertopic_output_", stage_label, ".csv")
  if (file.exists(csv_path)) {
    topic_info <- read.csv(csv_path)
    
    # Get top 25 topics (excluding -1 which is "outliers/unclassified")
    top_topics <- topic_info %>%
      filter(Topic != -1) %>%
      slice_max(Count, n = 25)
    
    # Plot the top topics
    print(
      ggplot(top_topics, aes(x = reorder(Name, Count), y = Count)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(title = paste("Top BERTopic Topics -", stage_name),
           x = "Topic Label", y = "Document Count")
    )
  } else {
    warning(paste("⚠️ File not found for", stage_label, "- please check if CSV was uploaded"))
  }
  
  # --- LDA Topic Modeling ---
  dtm <- tokens_df %>%
    mutate(document = 1) %>%
    count(document, word) %>%
    cast_dtm(document, word, n)
  
  if (nrow(dtm) > 0) {
    lda_model <- LDA(dtm, k = 5, control = list(seed = 1234))
    topics <- tidy(lda_model, matrix = "beta")
    
    top_terms <- topics %>%
      group_by(topic) %>%
      slice_max(beta, n = 10) %>%
      ungroup() %>%
      arrange(topic, -beta)
    
    lda_results[[stage_name]] <- top_terms  # Save results by stage name
    
  } else {
    warning(paste("⚠ Skipping LDA for", stage_name, "- DTM was empty"))
  }

}

# --- Ploting the LDA Topic Modeling ---
muted_colors1 <- c("#A3B18A", "#D9BF77", "#9EADC8", "#C6B89E", "#B4A284",
                  "#8E9AAF", "#C2B9B0", "#A9B4C2", "#CABFAB", "#91A6A6")

for (stage in names(lda_results)) {
  top_terms <- lda_results[[stage]]
  
  p <- ggplot(top_terms, aes(x = fct_reorder(term, beta), y = beta, fill = factor(topic))) +
    geom_col(show.legend = FALSE) +
    facet_wrap(~ topic, scales = "free") +
    coord_flip() +
    scale_fill_manual(values = muted_colors1) +
    labs(title = paste("Top LDA Topics -", stage),
         x = "Term", y = "Importance") +
    theme_minimal()
  
  print(p)
}
