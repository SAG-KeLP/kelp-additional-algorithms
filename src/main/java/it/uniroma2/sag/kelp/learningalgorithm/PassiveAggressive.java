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

package it.uniroma2.sag.kelp.learningalgorithm;

import java.util.Arrays;
import java.util.List;

import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.learningalgorithm.OnlineLearningAlgorithm;

/**
 * It is an online learning algorithms that implements the Passive Aggressive algorithms described in
 * 
 * [Crammer, JMLR2006] K. Crammer, O. Dekel, J. Keshet and S. Shalev-Shwartz. Online passive-aggressive algorithms. 
 * Journal of Machine Learning Research 7:551–585, 2006.
 * 
 * @author Simone Filice
 *
 */
public abstract class PassiveAggressive implements OnlineLearningAlgorithm, BinaryLearningAlgorithm{

	/**
	 * It is the updating policy applied by the Passive Aggressive Algorithm when a miss-prediction occurs
	 * 
	 * @author Simone Filice
	 */
	public enum Policy{
		/**
		 * The new prediction hypothesis after a new example \( \mathbf{x}_t\) with label \(y_t\) is observed is:  
		 * <p>
		 * \(argmin_{\mathbf{w}} \frac{1}{2} \left \| \mathbf{w}-\mathbf{w}_t \right \|^2\)   
		 * <p> such that \(  l(\mathbf{w};(\mathbf{x}_t,y_t))=0 \)
		 */
		HARD_PA,
		
		/**
		 * The new prediction hypothesis after a new example \( \mathbf{x}_t\) with label \(y_t\) is observed is:  
		 * <p>
		 * \(argmin_{\mathbf{w}} \frac{1}{2} \left \| \mathbf{w}-\mathbf{w}_t \right \|^2 + C\xi \)   
		 * <p>such that \(  l(\mathbf{w};(\mathbf{x}_t,y_t))\leq \xi  \)  and \( \xi\geq 0\)
		 */
		PA_I,
		
		/**
		 * The new prediction hypothesis after a new example \( \mathbf{x}_t\) with label \(y_t\) is observed is:  
		 * <p>
		 * \(argmin_{\mathbf{w}} \frac{1}{2} \left \| \mathbf{w}-\mathbf{w}_t \right \|^2 + C\xi^2 \)   
		 * <p>such that \(  l(\mathbf{w};(\mathbf{x}_t,y_t))\leq \xi  \)  and \( \xi\geq 0\)
		 */
		PA_II
	}
	
		
	protected Label label; 
	
	

	protected Policy policy = Policy.PA_II;

	protected float c = 1;//the aggressiveness parameter

	
	
	@Override
	public void reset() {
		this.getPredictionFunction().reset();
	}


	/**
	 * @return the updating policy
	 */
	public Policy getPolicy() {
		return policy;
	}


	/**
	 * @param policy the updating policy to set
	 */
	public void setPolicy(Policy policy) {
		this.policy = policy;
	}


	/**
	 * @return the aggressiveness parameter
	 */
	public float getC() {
		return c;
	}


	/**
	 * @param c the aggressiveness to set
	 */
	public void setC(float c) {
		this.c = c;
	}


	protected float computeWeight(Example example, float lossValue, float exampleSquaredNorm, float aggressiveness) {
		float weight=1;

		switch(policy){
		case HARD_PA:
			weight=lossValue/exampleSquaredNorm;
			break;
		case PA_I:
			weight=lossValue/exampleSquaredNorm;
			if(weight>aggressiveness){
				weight=aggressiveness;
			}
			break;
		case PA_II:
			weight=lossValue/(exampleSquaredNorm+1/(2*aggressiveness));
			break;
		}
	
		return weight;
	}
	
	
	@Override
	public void setLabels(List<Label> labels){
		if(labels.size()!=1){
			throw new IllegalArgumentException("The Passive Aggressive algorithm is a binary method which can learn a single Label");
		}
		else{
			this.label=labels.get(0);
			this.getPredictionFunction().setLabels(labels);
		}
	}


	@Override
	public List<Label> getLabels() {
		return Arrays.asList(label);
	}
	
	@Override
	public void learn(Dataset dataset){
		while(dataset.hasNextExample()){
			this.learn(dataset.getNextExample());
		}
		dataset.reset();
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
