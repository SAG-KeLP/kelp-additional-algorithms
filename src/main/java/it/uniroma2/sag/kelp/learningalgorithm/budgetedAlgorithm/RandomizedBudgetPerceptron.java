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

package it.uniroma2.sag.kelp.learningalgorithm.budgetedAlgorithm;

import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.learningalgorithm.BinaryLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.KernelMethod;
import it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.MetaLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.OnlineLearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.Prediction;
import it.uniroma2.sag.kelp.predictionfunction.PredictionFunction;
import it.uniroma2.sag.kelp.predictionfunction.model.BinaryKernelMachineModel;
import it.uniroma2.sag.kelp.predictionfunction.model.SupportVector;

import java.util.Random;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * It is a variation of the Randomized Budget Perceptron proposed in <p>
 * [CavallantiCOLT2006] G. Cavallanti, N. Cesa-Bianchi, C. Gentile. Tracking the best hyperplane with a simple budget Perceptron. In proc. of the 19-th annual conference on Computational Learning Theory. (2006)
 * <p>
 * Until the budget is not reached the online learning updating policy is the one of the baseAlgorithm that this
 * meta-algorithm is exploiting. When the budget is full, a random support vector is deleted and the perceptron updating policy is
 * adopted
 * 
 * @author Simone Filice
 *
 */
@JsonTypeName("randomizedPerceptron")
public class RandomizedBudgetPerceptron extends BudgetedLearningAlgorithm implements MetaLearningAlgorithm{

	private static final long DEFAULT_SEED=1;
	private long initialSeed = DEFAULT_SEED;
	@JsonIgnore
	private Random randomGenerator;
	
	private OnlineLearningAlgorithm baseAlgorithm;
	
	public RandomizedBudgetPerceptron(){
		randomGenerator = new Random(initialSeed);
	}
	
	public RandomizedBudgetPerceptron(int budget, OnlineLearningAlgorithm baseAlgorithm, long seed, Label label){
		randomGenerator = new Random(initialSeed);
		this.setBudget(budget);
		this.setBaseAlgorithm(baseAlgorithm);
		this.setSeed(seed);
		this.setLabel(label);		
	}
	
	/**
	 * Sets the seed for the random generator adopted to select the support vector to delete
	 * 
	 * @param seed the seed of the randomGenerator
	 */
	public void setSeed(long seed){
		this.initialSeed = seed;
		this.randomGenerator.setSeed(seed);
	}

	@Override
	public RandomizedBudgetPerceptron duplicate() {
		RandomizedBudgetPerceptron copy = new RandomizedBudgetPerceptron();
		copy.setBudget(budget);
		copy.setBaseAlgorithm(baseAlgorithm.duplicate());
		copy.setSeed(initialSeed);
		return copy;
	}

	@Override
	public void reset() {
		this.baseAlgorithm.reset();		
		this.randomGenerator.setSeed(initialSeed);
	}

	@Override
	protected Prediction predictAndLearnWithFullBudget(Example example) {
		Prediction prediction = this.baseAlgorithm.getPredictionFunction().predict(example);
		
		if((prediction.getScore(getLabel())>0) != example.isExampleOf(getLabel())){
			int svToDelete = this.randomGenerator.nextInt(budget);
			float weight = 1;
			if(!example.isExampleOf(getLabels().get(0))){
				weight=-1;
			}
			SupportVector sv = new SupportVector(weight, example);
			
			((BinaryKernelMachineModel)this.baseAlgorithm.getPredictionFunction().getModel()).setSupportVector(sv, svToDelete);
		}
		return prediction;
	}
	
	@Override
	public void setBaseAlgorithm(LearningAlgorithm baseAlgorithm) {
		if(baseAlgorithm instanceof OnlineLearningAlgorithm && baseAlgorithm instanceof KernelMethod && baseAlgorithm instanceof BinaryLearningAlgorithm){
			this.baseAlgorithm = (OnlineLearningAlgorithm) baseAlgorithm;
		}else{
			throw new IllegalArgumentException("a valid baseAlgorithm for the Randomized Budget Perceptron must implement OnlineLearningAlgorithm, BinaryLeaningAlgorithm and KernelMethod");
		}
	}

	@Override
	public OnlineLearningAlgorithm getBaseAlgorithm() {
		return this.baseAlgorithm;
	}
	
	@Override
	public PredictionFunction getPredictionFunction() {
		return this.baseAlgorithm.getPredictionFunction();
	}

	@Override
	public Kernel getKernel() {
		return ((KernelMethod)this.baseAlgorithm).getKernel();
	}

	@Override
	public void setKernel(Kernel kernel) {
		((KernelMethod)this.baseAlgorithm).setKernel(kernel);
		
	}

	@Override
	protected Prediction predictAndLearnWithAvailableBudget(Example example) {
		return this.baseAlgorithm.learn(example);
	}

}
