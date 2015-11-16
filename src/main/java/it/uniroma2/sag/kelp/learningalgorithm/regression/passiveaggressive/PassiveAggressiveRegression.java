/*
 * Copyright 2014 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.uniroma2.sag.kelp.learningalgorithm.regression.passiveaggressive;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.learningalgorithm.PassiveAggressive;
import it.uniroma2.sag.kelp.learningalgorithm.regression.RegressionLearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.regressionfunction.UnivariateRegressionOutput;
import it.uniroma2.sag.kelp.predictionfunction.regressionfunction.UnivariateRegressionFunction;

import com.fasterxml.jackson.annotation.JsonIgnore;

/**
 * Online Passive-Aggressive Learning Algorithm for regression tasks.
 *
 * reference: 
 * <p>
 * [CrammerJLMR2006] Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz and Yoram Singer
 * Online Passive-Aggressive Algorithms. Journal of Machine Learning Research (2006)
 * 
 * @author      Simone Filice
 */
public abstract class PassiveAggressiveRegression extends PassiveAggressive implements RegressionLearningAlgorithm{

	@JsonIgnore
	protected UnivariateRegressionFunction regressor;

	protected float epsilon;

	/**
	 * Returns epsilon, i.e. the accepted distance between the predicted and the real regression values
	 * 
	 * @return the epsilon
	 */
	public float getEpsilon() {
		return epsilon;
	}

	/**
	 * Sets epsilon, i.e. the accepted distance between the predicted and the real regression values
	 * 
	 * @param epsilon the epsilon to set
	 */
	public void setEpsilon(float epsilon) {
		this.epsilon = epsilon;
	}

	@Override
	public UnivariateRegressionFunction getPredictionFunction() {
		return this.regressor;
	}

	@Override
	public void learn(Dataset dataset){

		while(dataset.hasNextExample()){
			Example example = dataset.getNextExample();
			this.learn(example);
		}
		dataset.reset();
	}
	
	@Override
	public UnivariateRegressionOutput learn(Example example){
		UnivariateRegressionOutput prediction=this.regressor.predict(example);
		float difference = example.getRegressionValue(label) - prediction.getScore(label);
		float lossValue = Math.abs(difference) - epsilon;//it represents the distance from the correct semi-space
		if(lossValue>0){
			float exampleSquaredNorm = this.regressor.getModel().getSquaredNorm(example);
			float weight = this.computeWeight(example, lossValue, exampleSquaredNorm, c);
			if(difference<0){
				weight = -weight;
			}
			this.regressor.getModel().addExample(weight, example);			
		}		
		return prediction;
	}
	
}
