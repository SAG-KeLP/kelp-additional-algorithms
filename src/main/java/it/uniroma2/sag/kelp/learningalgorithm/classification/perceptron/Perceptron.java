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

import java.util.Arrays;
import java.util.List;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.learningalgorithm.BinaryLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.OnlineLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.ClassificationLearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryClassifier;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryMarginClassifierOutput;

import com.fasterxml.jackson.annotation.JsonIgnore;

/**
 * The perceptron learning algorithm algorithm for classification tasks. Reference:
 * <p> [Rosenblatt1957] F. Rosenblatt. The Perceptron – a perceiving and recognizing automaton. Report 85-460-1, Cornell Aeronautical Laboratory (1957)
 * 
 * @author Simone Filice
 *
 */
public abstract class Perceptron implements ClassificationLearningAlgorithm, OnlineLearningAlgorithm, BinaryLearningAlgorithm{

	@JsonIgnore
	protected BinaryClassifier classifier;

	protected Label label;

	protected float alpha=1;
	protected float margin = 1;
	protected boolean unbiased=false;

	/**
	 * Returns the learning rate, i.e. the weight associated to misclassified examples during the learning process
	 * 
	 * @return the learning rate
	 */
	public float getAlpha() {
		return alpha;
	}

	/**
	 * Sets the learning rate, i.e. the weight associated to misclassified examples during the learning process
	 * 
	 * @param alpha the learning rate to set
	 */
	public void setAlpha(float alpha) {
		if(alpha<=0 || alpha>1){
			throw new IllegalArgumentException("Invalid learning rate for the perceptron algorithm: valid alphas in (0,1]");
		}
		this.alpha = alpha;
	}

	/**
	 * Returns the desired margin, i.e. the minimum distance from the hyperplane that an example must have
	 * in order to be not considered misclassified  
	 * 
	 * @return the margin
	 */
	public float getMargin() {
		return margin;
	}

	/**
	 * Sets the desired margin, i.e. the minimum distance from the hyperplane that an example must have
	 * in order to be not considered misclassified  
	 * 
	 * @param margin the margin to set
	 */
	public void setMargin(float margin) {
		this.margin = margin;
	}

	/**
	 * Returns whether the bias, i.e. the constant term of the hyperplane, is always 0, or can be modified during
	 * the learning process
	 * 
	 * @return the unbiased
	 */
	public boolean isUnbiased() {
		return unbiased;
	}

	/**
	 * Sets whether the bias, i.e. the constant term of the hyperplane, is always 0, or can be modified during
	 * the learning process
	 * 
	 * @param unbiased the unbiased to set
	 */
	public void setUnbiased(boolean unbiased) {
		this.unbiased = unbiased;
	}


	@Override
	public void learn(Dataset dataset) {

		while(dataset.hasNextExample()){
			Example example = dataset.getNextExample();
			this.learn(example);
		}
		dataset.reset();
	}

	@Override
	public BinaryMarginClassifierOutput learn(Example example){
		BinaryMarginClassifierOutput prediction = this.classifier.predict(example);

		float predValue = prediction.getScore(label);
		if(Math.abs(predValue)<margin || prediction.isClassPredicted(label)!=example.isExampleOf(label) ){
			float weight = alpha;
			if(!example.isExampleOf(label)){
				weight = -alpha;
			}

			this.classifier.getModel().addExample(weight, example);
			if(!unbiased){
				float newBias = this.classifier.getModel().getBias() + weight;
				this.classifier.getModel().setBias(newBias);
			}
		}
		return prediction;
	}

	@Override
	public void reset() {
		this.classifier.reset();		
	}

	@Override
	public BinaryClassifier getPredictionFunction() {
		return this.classifier;
	}

	@Override
	public void setLabels(List<Label> labels){
		if(labels.size()!=1){
			throw new IllegalArgumentException("The Perceptron algorithm is a binary method which can learn a single Label");
		}
		else{
			this.label=labels.get(0);
			this.classifier.setLabels(labels);
		}
	}


	@Override
	public List<Label> getLabels() {
	
		return Arrays.asList(label);
	}
	
	@Override
	public Label getLabel(){
		return this.label;
	}
	
	@Override
	public void setLabel(Label label){
		this.setLabels(Arrays.asList(label));
	}

}
