import math
class NaiveBayesClassifier:
    def __init__(self, classes):
        self.classes = classes
        self.smoothingParameter = 0.1
        self.type = "naivebayes"
        self.tuning = False

    def setting_smoothing_parameter(self, parameter):
        self.smoothingParameter = parameter

    def fit(self, x_train, y_train, x_valid, y_valid):
        self.features = list(set([attribute for sample in x_train for attribute in sample.keys()]))
        
        self.numberOfFeatures = {}
        self.class_counts = {}
        self.featuresOfClass = {}

        for sample, cls in zip(x_train, y_train):
            if cls not in self.class_counts:
                self.class_counts[cls] = 0
            self.class_counts[cls] += 1
            
            for attribute, magnitude in sample.items():
                if (cls, attribute) not in self.featuresOfClass:
                    self.featuresOfClass[(cls, attribute)] = 0
                if magnitude > 0.01:
                    self.featuresOfClass[(cls, attribute)] += magnitude
                if attribute not in self.numberOfFeatures:
                    self.numberOfFeatures[attribute] = 0
                self.numberOfFeatures[attribute] += magnitude

        self.totals_in_class = {}
        for cls in self.classes:
            self.totals_in_class[cls] = sum(self.featuresOfClass.get((cls, attribute), 0) for attribute in self.features)

    def predict(self, test_samples):
        predictions = []
        self.logarithmicProbability_list = []
        
        for sample in test_samples:
            logarithmicProbability = self.calculatingLogarithmic_probability(sample)
            predictions.append(max(logarithmicProbability.items(), key=lambda x: x[1])[0])
            self.logarithmicProbability_list.append(logarithmicProbability)
            
        return predictions

    def calculatingLogarithmic_probability(self, sample):
        logarithmicProbability = {}
        amountOfClasses = sum(self.class_counts.values())

        for cls in self.classes:
            before = (self.class_counts[cls] + self.smoothingParameter) / (amountOfClasses + self.smoothingParameter * len(self.classes))
            presentLogarithmic_val = math.log(before)
            for attribute, magnitude in sample.items():
                if magnitude > 0.01:
                    feature_probabilities = (self.featuresOfClass.get((cls, attribute), 0) + self.smoothingParameter) / (self.totals_in_class[cls] + self.smoothingParameter * 2)
                    presentLogarithmic_val += math.log(max(feature_probabilities, 1e-10))
                else:
                    feature_probabilities = 1 - (self.featuresOfClass.get((cls, attribute), 0) + self.smoothingParameter) / (self.totals_in_class[cls] + self.smoothingParameter * 2)
                    presentLogarithmic_val += math.log(max(feature_probabilities, 1e-10))

            logarithmicProbability[cls] = presentLogarithmic_val

        return logarithmicProbability

    def retriveFeatures(self, class1, class2):
        featuresProbabiltiies = []
        for attribute in self.features:
            firstClassProbabilities = (self.featuresOfClass.get((class1, attribute), 0) + self.smoothingParameter) / (self.totals_in_class[class1] + self.smoothingParameter * 2)
            secondClassProbabilities = (self.featuresOfClass.get((class2, attribute), 0) + self.smoothingParameter) / (self.totals_in_class[class2] + self.smoothingParameter * 2)
            proportion = firstClassProbabilities / (secondClassProbabilities + 1e-10)
            featuresProbabiltiies.append((attribute, proportion))
            
        features_sorted = sorted(featuresProbabiltiies, key=lambda x: -x[1])
        return [attribute for attribute, _ in features_sorted[:100]]

if __name__ == "__main__":
    classifier = NaiveBayesClassifier(classes=["class1", "class2"])
    print("initialized")