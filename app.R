library(shiny)
library(bslib)
library(DT)
library(e1071)
library(caret)
library(ggplot2)
library(rsconnect)


ui <- page_navbar(
  title = "Naive Bayes Classification",
  theme = bs_theme(version = 5, bootswatch = "darkly"),
  
  nav_panel("Upload & Model",
            layout_sidebar(
              sidebar = sidebar(
                width = 320,
                fileInput("file", "Upload CSV", accept = ".csv"),
                selectInput("target", "Select the Target Variable", choices = NULL),
                selectInput("feature", "Select Feature to Plot", choices = NULL),
                hr(),
                strong("Select the Categorical Variables:"),
                div(style = "column-count: 2; column-gap: 10px;",
                    uiOutput("type_checkboxes")
                ),
                hr(),
                sliderInput("split_ratio", "% for the training set split", 
                            min = 50, max = 95, value = 80, step = 5, post = "%"),
                checkboxInput("set_seed", "Use fixed random seed (reproducible split)", value = TRUE),
                uiOutput("split_summary"),
                hr(),
                actionButton('train', 'Train the Model', class = "btn-primary")
              ),
              layout_column_wrap(
                width = 1,
                card(card_header('Dataset Preview'), DTOutput("table")),
                card(card_header("Target vs Feature Plot"), plotOutput("featureplot"))
              )
            )
  ),
  
  nav_panel("Performance",
            layout_column_wrap(
              width = 1, 
              card(card_header("Key Metrics"),
                   layout_column_wrap(
                     width = 1/4,
                     value_box(title = "Accuracy", value = textOutput("acc_val"), theme = "primary"),
                     value_box(title = "Precision", value = textOutput("precision_val"), theme = "secondary"),
                     value_box(title = "Recall", value = textOutput("recall_val"), theme = "info"),
                     value_box(title = "F1 Score", value = textOutput("f1_val"), theme = "success")
                   )
              ),
              layout_column_wrap(
                width = 1/2,
                card(card_header("Confusion Matrix"), plotOutput("cm_plot")),
                card(card_header("Per Class Metrics"), DTOutput("metrics_table"))
              )
            )
  ),
  
  nav_panel("Predictions",
            layout_sidebar(
              sidebar = sidebar(
                width = 320,
                strong("Predict on new data"),
                p(class = "text-muted", style = "font-size: 0.85em;",
                  "Train a model on the first tab before using this panel."),
                hr(),
                radioButtons("predict_mode", "Prediction mode",
                             choices = c("Upload CSV" = "csv", "Manual entry" = "manual"),
                             selected = "csv"),
                hr(),
                conditionalPanel(
                  condition = "input.predict_mode == 'csv'",
                  fileInput("predict_file", "Upload CSV to score (no target column needed)", accept = ".csv"),
                  actionButton("predict_csv_btn", "Predict", class = "btn-primary"),
                  br(), br(),
                  downloadButton("download_predictions", "Download Predictions")
                ),
                conditionalPanel(
                  condition = "input.predict_mode == 'manual'",
                  uiOutput("manual_inputs"),
                  actionButton("predict_manual_btn", "Predict", class = "btn-primary")
                )
              ),
              layout_column_wrap(
                width = 1,
                card(card_header("Prediction Result"), uiOutput("manual_result")),
                card(card_header("Scored Data"), DTOutput("predict_table"))
              )
            )
  )
)

server <- function(input, output, session) {
  
  data <- reactive({
    req(input$file)
    read.csv(input$file$datapath)
  })
  
  observeEvent(data(), {
    updateSelectInput(session, "target", choices = names(data()))
    updateSelectInput(session, "feature", choices = names(data()))
  })
  
  auto_is_categorical <- function(x) {
    is.character(x) || is.factor(x) || is.logical(x)
  }
  
  output$type_checkboxes <- renderUI({
    req(data())
    df <- data()
    defaults <- sapply(df, auto_is_categorical)
    
    checkboxGroupInput(
      "manual_categorical",
      label = NULL,
      choices = names(df),
      selected = names(df)[defaults]
    )
  })
  
  is_categorical_final <- function(colname) {
    colname %in% input$manual_categorical
  }
  
  output$split_summary <- renderUI({
    req(data(), input$split_ratio)
    n <- nrow(data())
    n_train <- round(n * input$split_ratio / 100)
    n_test <- n - n_train
    tags$small(
      style = "color: #aaa;",
      sprintf("≈ %d row train / %d row test (of %d total)", n_train, n_test, n)
    )
  })
  
  model_results <- eventReactive(input$train, {
    df <- data()
    target <- input$target
    cat_cols <- input$manual_categorical
    
    for (col in names(df)) {
      if (is_categorical_final(col)) {
        df[[col]] <- as.factor(df[[col]])
      }
    }
    
    if (input$set_seed) set.seed(42)
    
    index <- createDataPartition(df[[target]], p = input$split_ratio / 100, list = FALSE)
    train <- df[index, ]
    test <- df[-index, ]
    
    fit <- naiveBayes(as.formula(paste(target, "~ .")), data = train)
    preds <- predict(fit, test)
    
    list(
      cm = confusionMatrix(preds, test[[target]]),
      fit = fit,
      target = target,
      train = train,
      cat_cols = cat_cols,
      feature_cols = setdiff(names(train), target)
    )
  })
  
  output$table <- renderDT({ data() })
  
  output$featureplot <- renderPlot({
    req(data(), input$target, input$feature, input$manual_categorical)
    df <- data()
    target <- input$target
    feature <- input$feature
    
    req(target %in% names(df), feature %in% names(df))
    df[[target]] <- as.factor(df[[target]])
    
    if (is_categorical_final(feature)) {
      df[[feature]] <- as.factor(df[[feature]])
      ggplot(df, aes(x = .data[[feature]], fill = .data[[target]])) +
        geom_bar(position = "dodge") +
        labs(title = paste(feature, "vs", target), x = feature, y = "count") + 
        theme_minimal()
    } else {
      ggplot(df, aes(x = .data[[target]], y = .data[[feature]], fill = .data[[target]])) +
        geom_boxplot() + 
        labs(title = paste(feature, "vs", target), x = target, y = feature) + 
        theme_minimal() + theme(legend.position = "none")
    }
  })
  
  output$cm_plot <- renderPlot({
    req(model_results())
    cm <- model_results()$cm
    cm_df <- as.data.frame(cm$table)
    names(cm_df) <- c("Prediction", "Reference", "Freq")
    
    ggplot(cm_df, aes(y = Prediction, x = Reference, fill = Freq)) + 
      geom_tile(color = "white") + 
      geom_text(aes(label = Freq), color = "black", size = 5, fontface = "bold") + 
      scale_fill_gradient(low = "#e0f3ff", high = "#08519c") +
      labs(title = "Confusion Matrix", x = "Actual", y = "Predicted", fill = "Count") + 
      theme_minimal(base_size = 13) + 
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank()
      )
  })
  
  compute_per_class_metrics <- function(cm) {
    tbl <- cm$table
    classes <- rownames(tbl)
    total <- sum(tbl)
    
    do.call(rbind, lapply(classes, function(cls) {
      TP <- tbl[cls, cls]
      FP <- sum(tbl[cls, ]) - TP
      FN <- sum(tbl[, cls]) - TP
      TN <- total - TP - FP - FN
      support <- sum(tbl[, cls])
      
      precision   <- if ((TP + FP) == 0) NA else TP / (TP + FP)
      recall      <- if ((TP + FN) == 0) NA else TP / (TP + FN)
      specificity <- if ((TN + FP) == 0) NA else TN / (TN + FP)
      npv         <- if ((TN + FN) == 0) NA else TN / (TN + FN)
      f1          <- if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
        NA
      } else {
        2 * precision * recall / (precision + recall)
      }
      balanced_acc <- if (is.na(recall) || is.na(specificity)) NA else (recall + specificity) / 2
      prevalence   <- support / total
      
      data.frame(
        Class = cls,
        Precision = precision,
        Recall_Sensitivity = recall,
        Specificity = specificity,
        F1 = f1,
        Balanced_Accuracy = balanced_acc,
        NPV = npv,
        Prevalence = prevalence,
        Support = support,
        stringsAsFactors = FALSE
      )
    }))
  }
  
  output$acc_val <- renderText({
    req(model_results())
    sprintf("%.1f%%", model_results()$cm$overall["Accuracy"] * 100)
  })
  
  output$precision_val <- renderText({
    req(model_results())
    m <- compute_per_class_metrics(model_results()$cm)
    sprintf("%.1f%%", mean(m$Precision, na.rm = TRUE) * 100)
  })
  
  output$recall_val <- renderText({
    req(model_results())
    m <- compute_per_class_metrics(model_results()$cm)
    sprintf("%.1f%%", mean(m$Recall_Sensitivity, na.rm = TRUE) * 100)
  })
  
  output$f1_val <- renderText({
    req(model_results())
    m <- compute_per_class_metrics(model_results()$cm)
    sprintf("%.1f%%", mean(m$F1, na.rm = TRUE) * 100)
  })
  
  output$metrics_table <- renderDT({
    req(model_results())
    metrics_df <- compute_per_class_metrics(model_results()$cm)
    
    num_cols <- c("Precision", "Recall_Sensitivity", "Specificity", "F1",
                  "Balanced_Accuracy", "NPV", "Prevalence")
    metrics_df[num_cols] <- lapply(metrics_df[num_cols], function(x) round(x, 3))
    
    datatable(metrics_df, options = list(scrollX = TRUE, pageLength = 5), rownames = FALSE)
  })
  
  align_to_training <- function(newdata, train, feature_cols, cat_cols) {
    for (col in feature_cols) {
      if (!(col %in% names(newdata))) next
      if (col %in% cat_cols) {
        train_levels <- levels(as.factor(train[[col]]))
        newdata[[col]] <- factor(as.character(newdata[[col]]), levels = train_levels)
      } else {
        newdata[[col]] <- as.numeric(newdata[[col]])
      }
    }
    newdata
  }
  
  output$manual_inputs <- renderUI({
    req(model_results())
    r <- model_results()
    train <- r$train
    feature_cols <- r$feature_cols
    
    inputs <- lapply(feature_cols, function(col) {
      inputId <- paste0("pred_", col)
      if (col %in% r$cat_cols) {
        selectInput(inputId, col, choices = levels(as.factor(train[[col]])))
      } else {
        numericInput(inputId, col, value = round(mean(train[[col]], na.rm = TRUE), 2))
      }
    })
    do.call(tagList, inputs)
  })
  
  manual_prediction <- eventReactive(input$predict_manual_btn, {
    req(model_results())
    r <- model_results()
    feature_cols <- r$feature_cols
    
    row <- lapply(feature_cols, function(col) {
      val <- input[[paste0("pred_", col)]]
      req(!is.null(val))
      val
    })
    names(row) <- feature_cols
    newdata <- as.data.frame(row, stringsAsFactors = FALSE)
    newdata <- align_to_training(newdata, r$train, feature_cols, r$cat_cols)
    
    pred_class <- predict(r$fit, newdata, type = "class")
    pred_prob  <- predict(r$fit, newdata, type = "raw")
    
    list(class = pred_class, prob = pred_prob)
  })
  
  output$manual_result <- renderUI({
    if (input$predict_mode != "manual") {
      return(tags$p(class = "text-muted", "Switch to 'Manual entry' mode in the sidebar to use this."))
    }
    req(manual_prediction())
    res <- manual_prediction()
    prob_df <- as.data.frame(res$prob)
    
    tagList(
      h4(sprintf("Predicted Class: %s", as.character(res$class))),
      renderTable({
        data.frame(Class = colnames(prob_df),
                   Probability = sprintf("%.1f%%", as.numeric(prob_df[1, ]) * 100))
      })
    )
  })
  
  csv_predictions <- eventReactive(input$predict_csv_btn, {
    req(input$predict_file, model_results())
    r <- model_results()
    
    newdata <- read.csv(input$predict_file$datapath)
    req(all(r$feature_cols %in% names(newdata)))
    
    newdata <- align_to_training(newdata, r$train, r$feature_cols, r$cat_cols)
    
    pred_class <- predict(r$fit, newdata, type = "class")
    pred_prob  <- predict(r$fit, newdata, type = "raw")
    
    out <- cbind(newdata, Predicted = pred_class, round(as.data.frame(pred_prob), 3))
    out
  })
  
  output$predict_table <- renderDT({
    if (input$predict_mode != "csv") {
      return(datatable(data.frame(Message = "Switch to 'Upload CSV' mode in the sidebar to use this.")))
    }
    req(csv_predictions())
    datatable(csv_predictions(), options = list(scrollX = TRUE, pageLength = 8), rownames = FALSE)
  })
  
  output$download_predictions <- downloadHandler(
    filename = function() paste0("predictions_", Sys.Date(), ".csv"),
    content = function(file) {
      write.csv(csv_predictions(), file, row.names = FALSE)
    }
  )
}

shinyApp(ui, server)
