# TensorFlow_Advanced_Techniques_Specialization

High bias --> underfitting
High variance ---> overfitting

------------------------------------------------------
Train set error --> 1%  	|
							| 	-----> high variance
Dev Set Error ----> 11% 	|
------------------------------------------------------
Train set error --> 15%  	|
							| 	-----> high bias
Dev Set Error ----> 16% 	|
------------------------------------------------------
Train set error --> 15%  	|
							| 	-----> high variance and high bias
Dev Set Error ----> 30% 	|
------------------------------------------------------
Train set error --> 0.5%  	|
							| 	-----> low variance and low bias
Dev Set Error ----> 1% 	|
------------------------------------------------------



The functional API offers more flexibility and control over the layers than the sequential API. 

It can be used to predict multiple outputs(i.e output layers) with multiple inputs(i.e input layers))


try:
  10/0
except Exception:
  pass
  

plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

---------------------------
pd.Index(y_train).value_counts() --> value counts with array

Branch layer models eg --> Inception

contratistive loss --> ?


-----------------------------------------

model.compile(optimizer=rms, 
              loss = {'wine_type' : 'binary_crossentropy',
                       'wine_quality' : 'mean_squared_error'
                      },
              metrics = {'wine_type' : 'accuracy',
                          'wine_quality': tf.keras.metrics.RootMeanSquaredError()
                        }
              )
			  
			  