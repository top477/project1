export const FoundationQuiz = {
    topic: 'Machine Learning',
    level: 'foundation',
    totalQuestion: 20,
    perQuestionScore: 5,
    questions: [  
      {
        question: 'What is the first step in getting started with machine learning?',
        choices: ['Collecting data', 'Defining goals', 'Building models', 'Evaluating results'],
        type: 'MCQs',
        correctAnswer: 'Defining goals',
      },
      {
        question: 'Which step comes first in the machine learning process?',
        choices: ['Model selection', 'Data collection and preparation', 'Model training and evaluation', 'Feature engineering'],
        type: 'MCQs',
        correctAnswer: 'Data collection and preparation',
      },
      {
          question:
          'Probability theory is essential in machine learning for:',
          choices: ['Data preprocessing', 'Model evaluation', 'Feature engineering', 'Handling uncertainty and randomness'],
          type: 'MCQs',
          correctAnswer: 'Handling uncertainty and randomness',
      },
      {
          question: 'What does hypothesis testing involve in statistical methods?',
          choices: ['Estimating parameters of a distribution', 'Comparing sample means or proportions', 'Constructing confidence intervals', 'Performing feature selection'],
          type: 'MCQs',
          correctAnswer: 'Comparing sample means or proportions',
      },
      {
        question:
          'Which mathematical concept is fundamental to understanding vector operations in machine learning?',
        choices: ['Scalars', 'Matrices', 'Vectors', 'Eigenvalues'],
        type: 'MCQs',
        correctAnswer: 'Vectors',
      },
      {
        question: 'What is the primary goal of optimization in machine learning?',
        choices: ['Maximizing accuracy', 'Minimizing loss or error', 'Maximizing interpretability', 'Minimizing computational complexity'],
        type: 'MCQs',
        correctAnswer: 'Minimizing loss or error',
      },
      {
          question:
          'In calculus, what does the derivative of a function represent?',
          choices: ['The slope of the tangent line', 'The area under the curve', 'The limit of the function', 'The integral of the function'],
          type: 'MCQs',
          correctAnswer: 'The slope of the tangent line',
      },
      {
          question: 'What comes after defining goals in the machine learning process',
          choices: ['Collecting data', 'Model selection', 'Exploratory data analysis', 'Model deployment'],
          type: 'MCQs',
          correctAnswer: 'Collecting data',
      },
      {
        question: 'What is a key task in the data collection and preparation step?',
        choices: ['Model deployment', 'Data visualization', 'Model training', 'Feature selection'],
        type: 'MCQs',
        correctAnswer: 'Data visualization',
      },
      {
        question: 'What is the probability of an event that is certain to happen?',
        choices: ['0', '1', '0.5', '-1'],
        type: 'MCQs',
        correctAnswer: '1',
      },
      {
        question: 'Which statistical measure describes the spread or dispersion of data points?',
        choices: ['Mean', 'Median', 'Variance', 'Mode'],
        type: 'MCQs',
        correctAnswer: 'Variance',
      },
      {
        question: 'Which operation involves finding the dot product of two vectors?',
        choices: ['Matrix multiplication', 'Matrix addition', 'Scalar multiplication', 'Cross product'],
        type: 'MCQs',
        correctAnswer: 'Cross product',
      },
      {
          question: 'What does the learning rate control in optimization algorithms?',
          choices: ['The speed of convergence', 'The size of the dataset', 'The number of iterations', 'The complexity of the model'],
          type: 'MCQs',
          correctAnswer: 'The speed of convergence',
      },
      {
          question: 'What does the integral of a function represent geometrically?',
          choices: ['The slope of the tangent line', 'The area under the curve', 'The limit of the function', 'The derivative of the function'],
          type: 'MCQs',
          correctAnswer: 'The area under the curve',
      },
      {
        question: 'Which step involves understanding the problem and defining objectives in machine learning?',
        choices: ['Model deployment', 'Data collection and preparation', 'Model training and evaluation', 'Defining goals'],
        type: 'MCQs',
        correctAnswer: 'Defining goals',
      },
      {
        question: 'What is a key task in the model training and evaluation step?',
        choices: ['Data visualization', 'Hyperparameter tuning', 'Feature selection', 'Data preprocessing'],
        type: 'MCQs',
        correctAnswer: 'Hyperparameter tuning',
      },
      {
          question:
          'What is the probability of the complement of an event?',
          choices: ['The probability of the event itself', '0', '1', 'The difference between 1 and the probability of the event'],
          type: 'MCQs',
          correctAnswer: 'The difference between 1 and the probability of the event',
      },
      {
          question: 'Which statistical measure is robust to outliers?',
          choices: ['Mean', 'Median', 'Mode', 'Standard deviation'],
          type: 'MCQs',
          correctAnswer: 'Median',
      },
      {
        question: 'What is the determinant of an identity matrix?',
        choices: ['0', '1', '-1', 'It varies depending on the size of the matrix'],
        type: 'MCQs',
        correctAnswer: '1',
      },
      {
        question: 'What is the objective of gradient descent optimization?',
        choices: ['Maximizing the cost function', 'Minimizing the cost function', 'Finding the global maximum', 'Balancing the bias-variance tradeoff'],
        type: 'MCQs',
        correctAnswer: 'Minimizing the cost function',
      },
]}

export const BeginnerQuiz = {
    topic: 'Machine Learning',
    level: 'beginner',
    totalQuestion: 20,
    perQuestionScore: 5,
    questions: [
      {
          "question": "Which of the following is a valid Python variable name?",
          "choices": ["1variable_name", "variableName", "variable name", "variable-name"],
          "type": "MCQs",
          "correctAnswer": "variableName"
      },
      {
          "question": "Which machine learning algorithm is used for classification tasks and works by finding the best linear decision boundary between classes?",
          "choices": ["K-means clustering", "Linear regression", "Decision trees", "Support Vector Machines (SVM)"],
          "type": "MCQs",
          "correctAnswer": "Support Vector Machines (SVM)"
      },
      {
          "question": "Which of the following is a popular open-source software for machine learning and data mining?",
          "choices": ["Weka", "TensorFlow", "PyTorch", "scikit-learn"],
          "type": "MCQs",
          "correctAnswer": "Weka"
      },
      {
          "question": "Which Python library provides tools for data mining and data analysis, including various machine learning algorithms?",
          "choices": ["NumPy", "Pandas", "Matplotlib", "scikit-learn"],
          "type": "MCQs",
          "correctAnswer": "scikit-learn"
      },
      {
          "question": "Which R package provides a unified interface for various machine learning algorithms and facilitates model training, testing, and evaluation?",
          "choices": ["ggplot2", "dplyr", "caret", "tidyr"],
          "type": "MCQs",
          "correctAnswer": "caret"
      },
      {
          "question": "What is the primary objective of time series forecasting?",
          "choices": ["Predicting future events based on historical data", "Classifying data into categories", "Clustering similar data points", "Summarizing data distribution"],
          "type": "MCQs",
          "correctAnswer": "Predicting future events based on historical data"
      },
      {
          "question": "Which of the following is a common step in data preparation?",
          "choices": ["Model training", "Model deployment", "Data cleaning", "Model evaluation"],
          "type": "MCQs",
          "correctAnswer": "Data cleaning"
      },
      {
          "question": "What does the len() function do in Python?",
          "choices": ["Calculates logarithm", "Computes the length of a string or list", "Returns the maximum value in a list", "Rounds a floating-point number"],
          "type": "MCQs",
          "correctAnswer": "Computes the length of a string or list"
      },
      {
          "question": "Which algorithm is used for regression tasks and predicts continuous numeric values?",
          "choices": ["Decision trees", "K-nearest neighbors (KNN)", "Logistic regression", "Random forests"],
          "type": "MCQs",
          "correctAnswer": "Decision trees"
      },
      {
          "question": "What is the main advantage of using Weka for machine learning tasks?",
          "choices": ["It supports only a limited number of algorithms", "It requires advanced programming skills", "It provides a user-friendly graphical interface", "It is primarily used for deep learning"],
          "type": "MCQs",
          "correctAnswer": "It provides a user-friendly graphical interface"
      },
      {
          "question": "Which scikit-learn module is used for model evaluation and selection?",
          "choices": ["sklearn.preprocessing", "sklearn.feature_extraction", "sklearn.model_selection", "sklearn.metrics"],
          "type": "MCQs",
          "correctAnswer": "sklearn.model_selection"
      },
      {
          "question": "Which function is used to train a machine learning model in the caret package?",
          "choices": ["fit()", "train()", "predict()", "evaluate()"],
          "type": "MCQs",
          "correctAnswer": "train()"
      },
      {
          "question": "What is seasonality in time series data?",
          "choices": ["The overall trend of the data", "The daily fluctuations in data", "The repetitive pattern at fixed intervals", "The random noise in data"],
          "type": "MCQs",
          "correctAnswer": "The repetitive pattern at fixed intervals"
      },
      {
          "question": "What is one way to handle missing values in a dataset?",
          "choices": ["Remove the entire row with missing values", "Replace missing values with the mean of the column", "Replace missing values with the mode of the column", "Ignore missing values during analysis"],
          "type": "MCQs",
          "correctAnswer": "Replace missing values with the mean of the column"
      },
      {
          "question": "What does the sorted() function do in Python?",
          "choices": ["Removes duplicates from a list", "Sorts a list in ascending order", "Returns the sum of a list", "Reverses the order of a list"],
          "type": "MCQs",
          "correctAnswer": "Sorts a list in ascending order"
      },
      {
          "question": "Which algorithm is suitable for handling non-linear relationships between features and target variables?",
          "choices": ["Linear regression", "Decision trees", "Logistic regression", "K-means clustering"],
          "type": "MCQs",
          "correctAnswer": "Decision trees"
      },
      {
          "question": "Which file format is commonly used to import datasets into Weka?",
          "choices": [".csv", ".txt", ".xls", ".json"],
          "type": "MCQs",
          "correctAnswer": ".csv"
      },
      {
          "question": "Which scikit-learn module is used for feature extraction and preprocessing?",
          "choices": ["sklearn.preprocessing", "sklearn.model_selection", "sklearn.feature_selection", "sklearn.metrics"],
          "type": "MCQs",
          "correctAnswer": "sklearn.preprocessing"
      },
      {
          "question": "Which function is used to make predictions using a trained model in the caret package?",
          "choices": ["fit()", "train()", "predict()", "evaluate()"],
          "type": "MCQs",
          "correctAnswer": "predict()"
      },
      {
          "question": "What is autocorrelation in time series analysis?",
          "choices": ["The correlation between different variables", "The correlation between successive observations", "The correlation between past and future values", "The correlation between outliers in the data"],
          "type": "MCQs",
          "correctAnswer": "The correlation between successive observations"
      }
]}

export const IntermediateQuiz = {
  topic: 'Machine Learning',
  level: 'beginner',
  totalQuestion: 20,
  perQuestionScore: 5,
  questions: [
    {
        "question": "Which of the following libraries is commonly used for implementing machine learning algorithms in Python?",
        "choices": ["Matplotlib", "Seaborn", "scikit-learn", "PyTorch"],
        "type": "MCQs",
        "correctAnswer": "scikit-learn"
    },
    {
        "question": "What is the main advantage of the XGBoost algorithm?",
        "choices": ["It is only suitable for small datasets", "It cannot handle missing values", "It is an ensemble method that combines multiple weak learners", "It is computationally less efficient compared to other algorithms"],
        "type": "MCQs",
        "correctAnswer": "It is an ensemble method that combines multiple weak learners"
    },
    {
        "question": "What is a common challenge in imbalanced classification problems?",
        "choices": ["Overfitting", "Underfitting", "Class imbalance", "Feature engineering"],
        "type": "MCQs",
        "correctAnswer": "Class imbalance"
    },
    {
        "question": "Which deep learning framework is known for its simplicity and ease of use, suitable for beginners?",
        "choices": ["Keras", "PyTorch", "TensorFlow", "Theano"],
        "type": "MCQs",
        "correctAnswer": "Keras"
    },
    {
        "question": "What is a key feature of PyTorch?",
        "choices": ["It provides dynamic computational graphs", "It is primarily used for symbolic programming", "It is built on top of Theano", "It has limited support for GPU acceleration"],
        "type": "MCQs",
        "correctAnswer": "It provides dynamic computational graphs"
    },
    {
        "question": "Which of the following tasks can be performed using OpenCV in machine learning?",
        "choices": ["Object detection", "Text classification", "Speech recognition", "Graph-based clustering"],
        "type": "MCQs",
        "correctAnswer": "Object detection"
    },
    {
        "question": "What is the goal of better deep learning techniques?",
        "choices": ["Increasing model complexity", "Improving model interpretability", "Reducing training time", "Enhancing model performance and robustness"],
        "type": "MCQs",
        "correctAnswer": "Enhancing model performance and robustness"
    },
    {
        "question": "What is ensemble learning in machine learning?",
        "choices": ["Training multiple models separately and averaging their predictions", "Training a single model with multiple datasets", "Using a single learning algorithm to boost performance", "Utilizing multiple GPUs for parallel computation"],
        "type": "MCQs",
        "correctAnswer": "Training multiple models separately and averaging their predictions"
    },
    {
        "question": "Which of the following is NOT a step in implementing machine learning algorithms in code?",
        "choices": ["Data preprocessing", "Model evaluation", "Feature selection", "Model deployment"],
        "type": "MCQs",
        "correctAnswer": "Model deployment"
    },
    {
        "question": "What is one of the main advantages of using XGBoost over traditional gradient boosting?",
        "choices": ["XGBoost cannot handle missing values", "XGBoost is computationally slower", "XGBoost provides better regularization", "XGBoost is less flexible in handling different types of data"],
        "type": "MCQs",
        "correctAnswer": "XGBoost provides better regularization"
    },
    {
        "question": "Which technique can be used to handle class imbalance in classification tasks?",
        "choices": ["Overfitting", "Feature scaling", "Upsampling the minority class", "Decreasing the learning rate"],
        "type": "MCQs",
        "correctAnswer": "Upsampling the minority class"
    },
    {
        "question": "Which deep learning framework emphasizes dynamic computation graphs and imperative programming?",
        "choices": ["TensorFlow", "Keras", "PyTorch", "Theano"],
        "type": "MCQs",
        "correctAnswer": "PyTorch"
    },
    {
        "question": "What is a common application of ML in OpenCV?",
        "choices": ["Natural language processing", "Image processing and computer vision", "Speech recognition", "Time series analysis"],
        "type": "MCQs",
        "correctAnswer": "Image processing and computer vision"
    },
    {
        "question": "What is the primary focus of better deep learning techniques?",
        "choices": ["Improving hardware efficiency", "Enhancing model interpretability", "Optimizing loss functions", "Improving model performance and robustness"],
        "type": "MCQs",
        "correctAnswer": "Improving model performance and robustness"
    },
    {
        "question": "What is the main advantage of ensemble learning?",
        "choices": ["Simplifying model implementation", "Reducing computational complexity", "Improving model generalization and performance", "Increasing model interpretability"],
        "type": "MCQs",
        "correctAnswer": "Improving model generalization and performance"
    },
    {
        "question": "What is one of the key steps in training a machine learning model?",
        "choices": ["Feature deployment", "Model evaluation", "Data visualization", "Hyperparameter tuning"],
        "type": "MCQs",
        "correctAnswer": "Hyperparameter tuning"
    },
    {
        "question": "Which technique is commonly used for handling imbalanced datasets?",
        "choices": ["Random undersampling", "Data augmentation", "Feature scaling", "Increasing the learning rate"],
        "type": "MCQs",
        "correctAnswer": "Random undersampling"
    },
    {
        "question": "What distinguishes ensemble learning from traditional machine learning approaches?",
        "choices": ["Ensemble learning relies on unsupervised learning techniques", "Ensemble learning combines multiple models to improve performance", "Ensemble learning is more computationally intensive", "Ensemble learning requires larger datasets"],
        "type": "MCQs",
        "correctAnswer": "Ensemble learning combines multiple models to improve performance"
    },
    {
        "question": "What is the role of feature engineering in machine learning?",
        "choices": ["To deploy machine learning models", "To evaluate model performance", "To preprocess data for modeling", "To extract relevant features from raw data"],
        "type": "MCQs",
        "correctAnswer": "To extract relevant features from raw data"
    },
    {
        "question": "What is the purpose of regularization techniques in machine learning?",
        "choices": ["To increase model complexity", "To decrease model flexibility", "To reduce overfitting", "To amplify noise in the data"],
        "type": "MCQs",
        "correctAnswer": "To reduce overfitting"
    }
]}

export const AdvanceQuiz = {
  topic: 'Machine Learning',
  level: 'beginner',
  totalQuestion: 20,
  perQuestionScore: 5,
  questions: [
    {
        "question": "What is the main advantage of Long Short-Term Memory (LSTM) networks over traditional recurrent neural networks (RNNs)?",
        "choices": ["LSTMs can only handle short sequences of data", "LSTMs are less computationally efficient", "LSTMs can capture long-term dependencies in data", "LSTMs require fewer training iterations"],
        "type": "MCQs",
        "correctAnswer": "LSTMs can capture long-term dependencies in data"
    },
    {
        "question": "Which natural language processing task involves predicting the next word in a sequence given the previous words?",
        "choices": ["Text classification", "Named entity recognition", "Machine translation", "Language modeling"],
        "type": "MCQs",
        "correctAnswer": "Language modeling"
    },
    {
        "question": "What is the primary goal of computer vision?",
        "choices": ["Speech recognition", "Image classification", "Time series forecasting", "Natural language understanding"],
        "type": "MCQs",
        "correctAnswer": "Image classification"
    },
    {
        "question": "What is the main advantage of using convolutional neural networks (CNNs) for time series analysis?",
        "choices": ["CNNs automatically handle variable-length sequences", "CNNs can capture temporal dependencies in data", "CNNs are computationally less efficient", "CNNs are less susceptible to overfitting"],
        "type": "MCQs",
        "correctAnswer": "CNNs can capture temporal dependencies in data"
    },
    {
        "question": "What is a key application of Generative Adversarial Networks (GANs) in machine learning?",
        "choices": ["Image classification", "Time series analysis", "Data augmentation", "Speech recognition"],
        "type": "MCQs",
        "correctAnswer": "Data augmentation"
    },
    {
        "question": "What is the purpose of attention mechanisms in deep learning models?",
        "choices": ["To increase model complexity", "To reduce model interpretability", "To improve model performance on specific parts of the input", "To introduce randomness in model predictions"],
        "type": "MCQs",
        "correctAnswer": "To improve model performance on specific parts of the input"
    },
    {
        "question": "What is a key component of Transformer models used in natural language processing?",
        "choices": ["Convolutional layers", "Recurrent layers", "Attention mechanisms", "Pooling layers"],
        "type": "MCQs",
        "correctAnswer": "Attention mechanisms"
    },
    {
        "question": "What distinguishes Long Short-Term Memory (LSTM) networks from traditional recurrent neural networks (RNNs)?",
        "choices": ["LSTMs have a simpler architecture", "LSTMs can only process sequential data", "LSTMs can retain information over long sequences", "LSTMs are less susceptible to vanishing gradients"],
        "type": "MCQs",
        "correctAnswer": "LSTMs can retain information over long sequences"
    },
    {
        "question": "Which machine learning task involves processing and understanding human language?",
        "choices": ["Image classification", "Speech recognition", "Natural language processing", "Time series analysis"],
        "type": "MCQs",
        "correctAnswer": "Natural language processing"
    },
    {
        "question": "What is the primary objective of computer vision?",
        "choices": ["To analyze and interpret visual data", "To process and understand human language", "To recognize and classify speech signals", "To forecast future events based on historical data"],
        "type": "MCQs",
        "correctAnswer": "To analyze and interpret visual data"
    },
    {
        "question": "What is the primary advantage of using Convolutional Neural Networks (CNNs) for image classification?",
        "choices": ["They can process variable-length sequences", "They automatically extract relevant features from images", "They require fewer parameters compared to other models", "They are less computationally intensive"],
        "type": "MCQs",
        "correctAnswer": "They automatically extract relevant features from images"
    },
    {
        "question": "What is the main purpose of using Generative Adversarial Networks (GANs) in machine learning?",
        "choices": ["To improve model interpretability", "To generate realistic synthetic data", "To reduce computational complexity", "To perform dimensionality reduction"],
        "type": "MCQs",
        "correctAnswer": "To generate realistic synthetic data"
    },
    {
        "question": "What is the role of attention mechanisms in deep learning models?",
        "choices": ["To increase model complexity", "To reduce model interpretability", "To improve model performance on specific parts of the input", "To introduce randomness in model predictions"],
        "type": "MCQs",
        "correctAnswer": "To improve model performance on specific parts of the input"
    },
    {
        "question": "What is a key component of Transformer models used in natural language processing?",
        "choices": ["Convolutional layers", "Recurrent layers", "Attention mechanisms", "Pooling layers"],
        "type": "MCQs",
        "correctAnswer": "Attention mechanisms"
    },
    {
        "question": "What distinguishes Long Short-Term Memory (LSTM) networks from traditional recurrent neural networks (RNNs)?",
        "choices": ["LSTMs have a simpler architecture", "LSTMs can only process sequential data", "LSTMs can retain information over long sequences", "LSTMs are less susceptible to vanishing gradients"],
        "type": "MCQs",
        "correctAnswer": "LSTMs can retain information over long sequences"
    },
    {
        "question": "Which machine learning task involves processing and understanding human language?",
        "choices": ["Image classification", "Speech recognition", "Natural language processing", "Time series analysis"],
        "type": "MCQs",
        "correctAnswer": "Natural language processing"
    },
    {
        "question": "What is the primary objective of computer vision?",
        "choices": ["To analyze and interpret visual data", "To process and understand human language", "To recognize and classify speech signals", "To forecast future events based on historical data"],
        "type": "MCQs",
        "correctAnswer": "To analyze and interpret visual data"
    },
    {
        "question": "How can CNNs and LSTMs be combined for time series analysis?",
        "choices": ["CNN extracts features from each time step, then LSTM learns the temporal relationships", "CNN learns overall patterns, then LSTM predicts future values", "LSTM captures long-term trends, then CNN refines predictions for specific time points", "They cannot be effectively combined for time series tasks"],
        "type": "MCQs",
        "correctAnswer": "CNN extracts features from each time step, then LSTM learns the temporal relationships"
    },
    {
        "question": "What is a limitation of self-attention mechanisms in Transformers?",
        "choices": ["They cannot capture hierarchical relationships in text", "They require large amounts of labeled training data", "They are computationally expensive for long sequences", "They are not effective for tasks beyond natural language processing"],
        "type": "MCQs",
        "correctAnswer": "They are computationally expensive for long sequences."
    },
    {
        "question": "During training, what is the role of the discriminator in a Generative Adversarial Network (GAN)?",
        "choices": ["Provides labels for the training data", "Generates new data samples", "Evaluates the realism of generated data.s", "Optimizes the hyperparameters of the generator"],
        "type": "MCQs",
        "correctAnswer": "Generates new data samples"
    },
]}

export const AssessmentQuiz = {
  topic: 'Machine Learning Readiness Assessment Quiz',
  level: 'Readiness Assessment',
  totalQuestion: 10,
  questions: [
    {
      question: 'What is machine learning?',
      choices: [
        'A type of data entry', 
        'A way to teach computers to make decisions using data', 
        'A method for building websites', 
        'A tool for creating animations'
      ],
      correctAnswer: 'A way to teach computers to make decisions using data',
      type: 'MCQs',
    },
    {
      question: 'Which of these is a common type of machine learning?',
      choices: [
        'Automated drawing', 
        'Supervised learning', 
        'Video editing', 
        'Website hosting'
      ],
      correctAnswer: 'Supervised learning',
      type: 'MCQs',
    },
    {
      question: 'What is a dataset?',
      choices: [
        'A collection of data used for training models', 
        'A type of software application', 
        'A set of rules for coding', 
        'A group of web pages'
      ],
      correctAnswer: 'A collection of data used for training models',
      type: 'MCQs',
    },
    {
      question: 'What is the purpose of training a model in machine learning?',
      choices: [
        'To predict outcomes based on data', 
        'To create graphics', 
        'To format text', 
        'To manage files'
      ],
      correctAnswer: 'To predict outcomes based on data',
      type: 'MCQs',
    },
    {
      question: 'Which of these is an example of a machine learning model?',
      choices: [
        'Word processor', 
        'Linear regression', 
        'Web browser', 
        'Spreadsheet'
      ],
      correctAnswer: 'Linear regression',
      type: 'MCQs',
    },
    {
      question: 'What is overfitting in machine learning?',
      choices: [
        'A model that performs well on training data but poorly on new data', 
        'A type of graph', 
        'A way to clean data', 
        'A method for storing files'
      ],
      correctAnswer: 'A model that performs well on training data but poorly on new data',
      type: 'MCQs',
    },
    {
      question: 'What does "k" represent in k-means clustering?',
      choices: [
        'The number of clusters', 
        'The size of the dataset', 
        'The number of data points', 
        'The depth of a decision tree'
      ],
      correctAnswer: 'The number of clusters',
      type: 'MCQs',
    },
    {
      question: 'What is cross-validation used for?',
      choices: [
        'To evaluate the model’s performance on different data splits', 
        'To build a website', 
        'To edit a video', 
        'To write a program'
      ],
      correctAnswer: 'To evaluate the model’s performance on different data splits',
      type: 'MCQs',
    },
    {
      question: 'Which of these is used to evaluate a classification model?',
      choices: [
        'Accuracy', 
        'Pixels per inch', 
        'Page load time', 
        'Word count'
      ],
      correctAnswer: 'Accuracy',
      type: 'MCQs',
    },
    {
      question: 'What is a neural network?',
      choices: [
        'A model inspired by the human brain', 
        'A type of internet connection', 
        'A system for sending emails', 
        'A method for storing images'
      ],
      correctAnswer: 'A model inspired by the human brain',
      type: 'MCQs',
    },
  ],
};


  
  
