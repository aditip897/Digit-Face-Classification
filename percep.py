import numpy as np
class Perceptron:
    def __init__(self, valid_labels, maximum_num_iterations):
        self.valid_labels = valid_labels
        self.type = "perceptron"
        self.weights = {}
        self.bias = {}
        self.maximum_num_iterations = maximum_num_iterations
       
        for label in valid_labels:
            self.bias[label] = 0.0

            self.weights[label] = {}
    
    def weightSetter(self, weights):
        assert len(weights) == len(self.valid_labels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, testData, testLabels):
        
        learning_rate = 2.0
        highest_weight = None
        bias_value = None
        most_accurate = 0.0
        
        
        set_of_features = set()
        for datum in trainingData:
            set_of_features.update(datum.keys())
            
        for label in self.valid_labels:
            for feature in set_of_features:
                self.weights[label][feature] = np.random.normal(0, 0.01)
        
        for iter in range(self.maximum_num_iterations):
            errors = 0
            
            indexes = np.random.permutation(len(trainingData))
            
            for i in indexes:
                datum = trainingData[i]
                actual_label = trainingLabels[i]
                
                data_scores = {}
                for label in self.valid_labels:
                    score = self.bias[label]
                    sum_of_feat_vals = 0.0
                    for feature, value in datum.items():
                        sum_of_feat_vals += value * self.weights[label][feature]
                    data_scores[label] = score + sum_of_feat_vals / max(1.0, len(datum))
                
                pred_label = max(data_scores, key=data_scores.get)
                
                margin = 1.0  
                if pred_label != actual_label or (data_scores[pred_label] - data_scores[actual_label]) < margin:
                    errors += 1
                    
                    for feature, value in datum.items():
                        self.weights[actual_label][feature] += learning_rate * value
                        self.weights[pred_label][feature] -= learning_rate * value
                    
                    self.bias[pred_label] -= learning_rate
                    self.bias[actual_label] += learning_rate

            
            preds = self.classify(trainingData)
            current_accuracy = sum(1 for i in range(len(trainingLabels)) 
                                 if preds[i] == trainingLabels[i]) / len(trainingLabels)
            
            if current_accuracy > most_accurate:
                most_accurate = current_accuracy
                highest_weight = {label: self.weights[label].copy() for label in self.valid_labels}
                bias_value = self.bias.copy()
            
            
            if errors > len(trainingData) * 0.2: 
                learning_rate = min(2.0, learning_rate * 1.1) 
            else:
                learning_rate = max(0.1, learning_rate * 0.8)  
            
            if current_accuracy >= 0.85:
                break
        
        if highest_weight is not None:
            self.weights = highest_weight
            self.bias = bias_value

    def classify(self, data):
        guesses = []
        for datum in data:
            data_scores = {}
            for label in self.valid_labels:
                score = self.bias[label]
                sum_of_feat_vals = 0.0
                for feature, value in datum.items():
                    if feature in self.weights[label]:
                        sum_of_feat_vals += value * self.weights[label][feature]
                data_scores[label] = score + sum_of_feat_vals / max(1.0, len(datum))
            guesses.append(max(data_scores, key=data_scores.get))
        return guesses

    def findHighWeightFeatures(self, label):
        feature_weights = [(a, abs(b)) for a, b in self.weights[label].items()]
        feature_weights.sort(key=lambda x: -x[1])
        return [a for a, b in feature_weights[:100]]