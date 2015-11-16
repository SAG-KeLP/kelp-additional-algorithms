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

package it.uniroma2.sag.kelp.learningalgorithm.classification.perceptron;


import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.learningalgorithm.LinearMethod;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryLinearClassifier;
import it.uniroma2.sag.kelp.predictionfunction.model.BinaryLinearModel;

import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * The perceptron learning algorithm algorithm for classification tasks (linear version). Reference:
 * <p> [Rosenblatt1957] F. Rosenblatt. The Perceptron – a perceiving and recognizing automaton. Report 85-460-1, Cornell Aeronautical Laboratory (1957)
 * 
 * @author Simone Filice
 *
 */
@JsonTypeName("linearPerceptron")
public class LinearPerceptron extends Perceptron implements LinearMethod{

	
	private String representation;

	
	public LinearPerceptron(){
		this.classifier = new BinaryLinearClassifier();
		this.classifier.setModel(new BinaryLinearModel());
	}	
	
	public LinearPerceptron(float alpha, float margin, boolean unbiased, String representation, Label label){
		this.classifier = new BinaryLinearClassifier();
		this.classifier.setModel(new BinaryLinearModel());
		this.setAlpha(alpha);
		this.setMargin(margin);
		this.setUnbiased(unbiased);
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
	public LinearPerceptron duplicate(){
		LinearPerceptron copy = new LinearPerceptron();		
		copy.setAlpha(this.alpha);
		copy.setMargin(this.margin);		
		copy.setRepresentation(representation);
		copy.setUnbiased(this.unbiased);
		return copy;
	}
	
	@Override
	public BinaryLinearClassifier getPredictionFunction(){
		return (BinaryLinearClassifier) this.classifier;
	}

}
