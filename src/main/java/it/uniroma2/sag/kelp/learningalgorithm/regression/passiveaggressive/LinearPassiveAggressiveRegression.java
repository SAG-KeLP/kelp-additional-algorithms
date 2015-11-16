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

import com.fasterxml.jackson.annotation.JsonTypeName;

import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.learningalgorithm.LinearMethod;
import it.uniroma2.sag.kelp.predictionfunction.model.BinaryLinearModel;
import it.uniroma2.sag.kelp.predictionfunction.regressionfunction.UnivariateLinearRegressionFunction;

/**
 * Online Passive-Aggressive Learning Algorithm for regression tasks (linear version).
 *
 * reference: 
 * <p>
 * [CrammerJLMR2006] Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz and Yoram Singer
 * Online Passive-Aggressive Algorithms. Journal of Machine Learning Research (2006)
 * 
 * @author      Simone Filice
 */
@JsonTypeName("linearPA-R")
public class LinearPassiveAggressiveRegression extends PassiveAggressiveRegression implements LinearMethod{

	private String representation;
	
	public LinearPassiveAggressiveRegression(){
		UnivariateLinearRegressionFunction regressor = new UnivariateLinearRegressionFunction();
		regressor.setModel(new BinaryLinearModel());
		this.regressor = regressor;
		
	}
	
	public LinearPassiveAggressiveRegression(float aggressiveness, float epsilon, Policy policy, String representation, Label label){
		UnivariateLinearRegressionFunction regressor = new UnivariateLinearRegressionFunction();
		regressor.setModel(new BinaryLinearModel());
		this.regressor = regressor;
		this.setC(aggressiveness);
		this.setEpsilon(epsilon);
		this.setPolicy(policy);
		this.setRepresentation(representation);
		this.setLabel(label);
	}
	
	@Override
	public LinearPassiveAggressiveRegression duplicate() {
		LinearPassiveAggressiveRegression copy = new LinearPassiveAggressiveRegression();
		copy.setC(this.c);
		copy.setRepresentation(this.representation);
		copy.setPolicy(this.policy);
		copy.setEpsilon(epsilon);
		return copy;
	}

	@Override
	public String getRepresentation() {
		return representation;				
	}

	@Override
	public void setRepresentation(String representation) {
		this.representation = representation;
		this.getPredictionFunction().getModel().setRepresentation(representation);
	}

	@Override
	public UnivariateLinearRegressionFunction getPredictionFunction(){
		return (UnivariateLinearRegressionFunction) this.regressor;
	}
	
}
