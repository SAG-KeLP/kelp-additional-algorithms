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

package it.uniroma2.sag.kelp.learningalgorithm.classification.passiveaggressive;

import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.learningalgorithm.LinearMethod;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryLinearClassifier;
import it.uniroma2.sag.kelp.predictionfunction.model.BinaryLinearModel;

import com.fasterxml.jackson.annotation.JsonTypeName;


/**
 * Online Passive-Aggressive Learning Algorithm for classification tasks (linear version) .
 * Every time an example is misclassified it is added the the current hyperplane, with the weight that solves the 
 * passive aggressive minimization problem
 * 
 * reference: 
 * <p>
 * [CrammerJLMR2006] Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz and Yoram Singer
 * Online Passive-Aggressive Algorithms. Journal of Machine Learning Research (2006)
 * 
 * <p>The standard algorithm is modified, including the fairness extention from<p>
 * [FiliceECIR2014] S. Filice, G. Castellucci, D. Croce, and R. Basili. Effective Kernelized Online Learning 
 * in Language Processing Tasks. In collection of Advances in Information Retrieval, pp. 347–358, Springer International Publishing, 2014. 
 * 
 * @author      Simone Filice
 */
@JsonTypeName("linearPA")
public class LinearPassiveAggressiveClassification extends PassiveAggressiveClassification implements LinearMethod{
	
	private String representation;

	public LinearPassiveAggressiveClassification(){
		this.classifier = new BinaryLinearClassifier();
		this.classifier.setModel(new BinaryLinearModel());
	}	
	
	public LinearPassiveAggressiveClassification(float cp, float cn, Loss loss, Policy policy, String representation, Label label){
		this.classifier = new BinaryLinearClassifier();
		this.classifier.setModel(new BinaryLinearModel());
		this.setCp(cp);
		this.setCn(cn);
		this.setLoss(loss);
		this.setPolicy(policy);
		this.setRepresentation(representation);
		this.setLabel(label);
	}	

	@Override
	public String getRepresentation() {
		return representation;
	}

	@Override
	public void setRepresentation(String representation) {
		this.representation = representation;
		BinaryLinearModel model = (BinaryLinearModel) this.classifier.getModel();
		model.setRepresentation(representation);
	}

	@Override
	public LinearPassiveAggressiveClassification duplicate(){
		LinearPassiveAggressiveClassification copy = new LinearPassiveAggressiveClassification();		
		copy.setRepresentation(this.representation);
		copy.setCp(this.cp);
		copy.setCn(this.c);
		copy.setFairness(this.fairness);
		copy.setLoss(this.loss);
		copy.setPolicy(this.policy);
		return copy;
	}
	
	@Override
	public BinaryLinearClassifier getPredictionFunction(){
		return (BinaryLinearClassifier) this.classifier;
	}

}
