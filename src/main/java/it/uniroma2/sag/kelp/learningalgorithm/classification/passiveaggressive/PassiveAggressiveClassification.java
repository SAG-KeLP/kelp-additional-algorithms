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

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.learningalgorithm.PassiveAggressive;
import it.uniroma2.sag.kelp.learningalgorithm.classification.ClassificationLearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryClassifier;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryMarginClassifierOutput;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;

/**
 * Online Passive-Aggressive Learning Algorithm for classification tasks.
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
 * in Language Processing Tasks. In collection of Advances in Information Retrieval, pp. 347-358, Springer International Publishing, 2014. 
 * 
 * @author      Simone Filice
 */
public abstract class PassiveAggressiveClassification extends PassiveAggressive implements ClassificationLearningAlgorithm{

	public enum Loss{
		HINGE,
		RAMP
	}

	protected Loss loss = Loss.HINGE;
	protected float cp = c;//cp is the aggressiveness w.r.t. positive examples. c will be considered the aggressiveness w.r.t. negative examples
	protected boolean fairness = false;
	
	@JsonIgnore
	protected BinaryClassifier classifier;


	/**
	 * @return the fairness
	 */
	public boolean isFairness() {
		return fairness;
	}


	/**
	 * @param fairness the fairness to set
	 */
	public void setFairness(boolean fairness) {
		this.fairness = fairness;
	}

	/**
	 * @return the aggressiveness parameter for positive examples
	 */
	public float getCp() {
		return cp;
	}


	/**
	 * @param cp the aggressiveness parameter for positive examples
	 */
	public void setCp(float cp) {
		this.cp = cp;
	}
	
	/**
	 * @return the aggressiveness parameter for negative examples
	 */
	public float getCn() {
		return c;
	}


	/**
	 * @param cn the aggressiveness parameter for negative examples
	 */
	public void setCn(float cn) {
		this.c = cn;
	}
	
	@Override
	@JsonIgnore
	public float getC(){
		return c;
	}
	
	@Override
	@JsonProperty
	public void setC(float c){
		super.setC(c);
		this.cp=c;
	}

	/**
	 * @return the loss function type
	 */
	public Loss getLoss() {
		return loss;
	}


	/**
	 * @param loss the loss function type to set
	 */
	public void setLoss(Loss loss) {
		this.loss = loss;
	}

	@Override
	public BinaryClassifier getPredictionFunction() {
		return this.classifier;
	}

	@Override
	public BinaryMarginClassifierOutput learn(Example example){

		BinaryMarginClassifierOutput prediction=this.classifier.predict(example);

		float lossValue = 0;//it represents the distance from the correct semi-space
		if(prediction.isClassPredicted(label)!=example.isExampleOf(label)){
			lossValue = 1 + Math.abs(prediction.getScore(label));
		}else if(Math.abs(prediction.getScore(label))<1){
			lossValue = 1 - Math.abs(prediction.getScore(label));			
		}			

		if(lossValue>0 && (lossValue<2 || this.loss!=Loss.RAMP)){
			float exampleAggressiveness=this.c;
			if(example.isExampleOf(label)){
				exampleAggressiveness=cp;
			}
			float exampleSquaredNorm = this.classifier.getModel().getSquaredNorm(example);
			float weight = this.computeWeight(example, lossValue, exampleSquaredNorm ,exampleAggressiveness);
			if(!example.isExampleOf(label)){
				weight*=-1;
			}
			this.getPredictionFunction().getModel().addExample(weight, example);	
		}
		return prediction;

	}
	
	@Override
	public void learn(Dataset dataset){
		if(this.fairness){
			float positiveExample = dataset.getNumberOfPositiveExamples(label);
			float negativeExample = dataset.getNumberOfNegativeExamples(label);
			cp = c * negativeExample / positiveExample;
		}
		//System.out.println("cn: " + c + " cp: " + cp);
		super.learn(dataset);
	}
	
}
