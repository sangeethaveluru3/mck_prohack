class stacked_model_reg():
    
    #passthrough means that the second layer model uses the training data as well as the first layer predictions
    #use probability means that the second layer model uses model.predict_proba() rather than predicting classes instead
    def __init__(self,
                 base_models,
                 second_layer_model,
                 allow_passthrough=False):
        self.base_models = base_models
        self.second_layer_model = second_layer_model
        self.allow_passthrough = allow_passthrough
        self.first_layer_train_preds = None
        self.first_layer_test_preds = None

    #this function gets first level model predictions and stores them within the class object
    def first_layer_predict(self, X_train, X_test, y_train):
        '''get out of fold predictions from a model 
        for training set and test set'''

        #creating array to fit all the first layer models predictions in
        first_layer_training = np.zeros(
            (X_train.shape[0], len(self.base_models)))
        first_layer_test = np.zeros((X_test.shape[0], len(self.base_models)))

        
        for ind, clf in enumerate(self.base_models):
            #init what we are going to put our predictions into
            #predictions on training set
            oof_train = np.zeros((X_train.shape[0], ))
            #predictions on test set
            oof_test = np.zeros((X_test.shape[0], ))
            #aggregating the predictions
            oof_test_batch = np.empty((5, X_test.shape[0]))

            #now going to loop over folds in our dataset and fit a model on the training data
            #then we predict on the training set and the test set
            for i, (train_index, test_index) in enumerate(
                    kf.split(X_train, y_train.values)):

                #because x is a numpy array and y is a pandas series have to do iloc on y but not on x
                x_batch_train = X_train[train_index]
                y_batch_train = y_train.iloc[train_index]
                x_batch_test = X_train[test_index]

                #here we are training the classifier
                clf.fit(x_batch_train, y_batch_train)
                #to keep track of what stage of modeling we are at
                print(ind,i)

                #here we are training the classifier
                #storing predictions for the test batch of X_train (bit confusing terminology)
                oof_train[test_index] = clf.predict(x_batch_test)
                #storing predictions for X_test
                oof_test_batch[i, :] = clf.predict(X_test)
            #aggregating the batches
            oof_test[:] = oof_test_batch.mean(axis=0)
            #reshaping so they are columns instead of rows -easier to put back into a dataframe
            first_layer_training[:,
                                 ind] = np.ravel(oof_train.reshape(-1, 1))
            first_layer_test[:, ind] = np.ravel(oof_test.reshape(-1, 1))

            #now putting all the predictions for the first layer into self so can use the second layer for predictions
            if self.allow_passthrough == False:

                self.first_layer_train_preds = first_layer_training
                self.first_layer_test_preds = first_layer_test
                
            else:

                first_layer_train_data = np.concatenate(
                    (X_train, first_layer_training), axis=1)
                first_layer_test_data = np.concatenate(
                    (X_test, first_layer_test), axis=1)
                self.first_layer_train_preds = first_layer_train_data
                self.first_layer_test_preds = first_layer_test_data
                

            #now putting all the predictions for the first layer into self so can use the second layer for predictions
            if self.allow_passthrough == False:

                self.first_layer_train_preds = first_layer_training
                self.first_layer_test_preds = first_layer_test
                
            else:

                first_layer_train_data = np.concatenate(
                    (X_train, first_layer_training), axis=1)
                first_layer_test_data = np.concatenate(
                    (X_test, first_layer_test), axis=1)
                self.first_layer_train_preds = first_layer_train_data
                self.first_layer_test_preds = first_layer_test_data
                

    #fits second layer model onto X_train predictions
    def second_layer_fit(self, y_train):

        self.second_layer_model.fit(self.first_layer_train_preds, y_train)

    def get_first_layer_preds(self):
        return self.first_layer_train_preds

    #returns the cv scores and its mean for the model
    def cv_score(self, X_train, y_train, cv=5, shuffle=False):

        if X_train == None:
            cv_scores = cross_val_score(self.second_layer_model,
                                        self.first_layer_train_preds,
                                        y_train,
                                        cv=cv
                                       )
            return cv_scores, cv_scores.mean()
        else:
            cv_scores = cross_val_score(self.second_layer_model,
                                        X_train,
                                        y_train,
                                        cv=cv)
            return cv_scores, cv_scores.mean()

    #same as sklearn version basically
    def score(self, X, y):

        if X == 'train':
            return self.second_layer_model.score(self.first_layer_train_preds,
                                                 y)

        if X == 'test':
            return self.second_layer_model.score(self.first_layer_test_preds,
                                                 y)
        else:
            return self.second_layer_model.score(X, y)

    def predict(self, X='train'):

        if X == 'train':
            return self.second_layer_model.predict(
                self.first_layer_train_preds)

        if X == 'test':
            return self.second_layer_model.predict(self.first_layer_test_preds)

        else:
            return self.second_layer_model.predict(X)

    def predict_probs(self, X):
        return self.second_layer_model.predict_proba(X)

    def get_features(self):
        return self.second_layer_model.feature_importances_