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
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.budgetedAlgorithm.BudgetedLearningAlgorithm;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryKernelMachineClassifier;
import it.uniroma2.sag.kelp.predictionfunction.classifier.BinaryMarginClassifierOutput;
import it.uniroma2.sag.kelp.predictionfunction.model.BinaryKernelMachineModel;
import it.uniroma2.sag.kelp.predictionfunction.model.SupportVector;

import java.util.List;

import org.ejml.alg.dense.mult.VectorVectorMult;
import org.ejml.data.DenseMatrix64F;
import org.ejml.ops.CommonOps;

/**
 * Online Passive-Aggressive on a budget 
 * reference: Zhuang Wang and Slobodan Vucetic
 * Online Passive-Aggressive Algorithms on a Budget
 * 
 * @author      Simone Filice
 */
public class BudgetedPassiveAggressiveClassification extends BudgetedLearningAlgorithm{

	/**
	 * It is the updating policy applied when the budget is full.
	 * 
	 * @author Simone Filice
	 */
	public enum DeletingPolicy{
		
		/**
		 * Budgeted Passive Aggressive Simple: when a new support vector must be added, one is removed and the weight
		 * of the other support vectors is kept unchanged
		 */
		BPA_S,
		
		/**
		 * Budgeted Passive Aggressive Nearest Neighbor: when a new support vector must be added, one is removed and the weight
		 * of its nearest neighbor is adjusted
		 */
		BPA_1NN
	}

	private DeletingPolicy deletingPolicy = DeletingPolicy.BPA_S;
	private Kernel kernel;
	private BinaryKernelMachineClassifier classifier;
	private boolean fairness = false;
	private float cp = 1;
	private float cn = 1;

	private boolean areNnComputed = false;
	private int [] nearestNeighbors = null;//used only with BPA_1NN_DELETING_POLICY 
	private float [] nearestNeighborsSimilarity = null;//used only with BPA_1NN_DELETING_POLICY

	/*
	 * The following DenseMatrix64F are used during the 1NN updating policy. 
	 * They are pre-allocated in order to avoid waste of memory resouces
	 */
	private DenseMatrix64F krVector = new DenseMatrix64F(2,1);//krVector = k_r
	private DenseMatrix64F beta = new DenseMatrix64F(2, 1);
	private DenseMatrix64F kMatrix = new DenseMatrix64F(2,2);
	private DenseMatrix64F ktVector = new DenseMatrix64F(2,1); //ktVector = k_t
	private DenseMatrix64F Kkr = new DenseMatrix64F(2,1);//Kkr = K^1 * k_r  
	private DenseMatrix64F Kkt = new DenseMatrix64F(2,1);//Kkt = K^1 * k_t
	
	public BudgetedPassiveAggressiveClassification(){
		this.classifier = new BinaryKernelMachineClassifier();
		this.classifier.setModel(new BinaryKernelMachineModel());
	}

	public BudgetedPassiveAggressiveClassification(int budget, Kernel kernel, float cp, float cn, DeletingPolicy deletingPolicy, Label label){
		this();
		this.setDeletingPolicy(deletingPolicy);
		this.setCn(cn);
		this.setCp(cp);
		this.setKernel(kernel);
		this.setBudget(budget);
		this.setLabel(label);
	}

	public BudgetedPassiveAggressiveClassification(int budget, Kernel kernel, float c, boolean fairness, DeletingPolicy deletingPolicy, Label label){
		this();
		this.setDeletingPolicy(deletingPolicy);
		this.setCn(c);
		this.setCp(c);
		this.setFairness(fairness);
		this.setKernel(kernel);
		this.setBudget(budget);
		this.setLabel(label);
	}

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
		return cn;
	}


	/**
	 * @param cn the aggressiveness parameter for negative examples
	 */
	public void setCn(float cn) {
		this.cn = cn;
	}

	/**
	 * @param c the aggressiveness parameter
	 */
	public void setC(float c) {
		this.cn = c;
		this.cp = c;
	}

	/**
	 * @return the deletingPolicy
	 */
	public DeletingPolicy getDeletingPolicy() {
		return deletingPolicy;
	}

	/**
	 * @param deletingPolicy the deletingPolicy to set
	 */
	public void setDeletingPolicy(DeletingPolicy deletingPolicy) {
		this.deletingPolicy = deletingPolicy;
	}

	@Override
	public LearningAlgorithm duplicate() {
		BudgetedPassiveAggressiveClassification copy = new BudgetedPassiveAggressiveClassification(budget, kernel, cp, cn, deletingPolicy, label);
		copy.setFairness(fairness);
		return copy;
	}

	@Override
	public void reset() {
		this.classifier.reset();
		this.areNnComputed = false;
	}

	@Override
	public BinaryKernelMachineClassifier getPredictionFunction() {
		return this.classifier;
	}

	@Override
	public Kernel getKernel() {
		return kernel;
	}

	@Override
	public void setKernel(Kernel kernel) {
		this.kernel = kernel;
		this.getPredictionFunction().getModel().setKernel(kernel);
	}

	@Override
	protected BinaryMarginClassifierOutput predictAndLearnWithAvailableBudget(Example example) {
		BinaryMarginClassifierOutput prediction=this.classifier.predict(example);

		float lossValue = this.evaluateLoss(prediction.getScore(label), example);
		if(lossValue>0){
			float exampleAggressiveness=this.cn;
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

	private float computeWeight(Example example, float lossValue, float exampleSquaredNorm, float aggressiveness) {
		float weight=lossValue/exampleSquaredNorm;
		if(weight>aggressiveness){
			weight=aggressiveness;
		}

		return weight;
	}

	@Override
	protected BinaryMarginClassifierOutput predictAndLearnWithFullBudget(Example example) {
		BinaryMarginClassifierOutput prediction=this.classifier.predict(example);

		float lossValue = evaluateLoss(prediction.getScore(label), example);			

		if(lossValue>0){
			float exampleAggressiveness=this.cn;
			if(example.isExampleOf(label)){
				exampleAggressiveness=cp;
			}

			switch (this.deletingPolicy) {
			case BPA_S:
				this.bpaSDeletingPolicy(example, exampleAggressiveness, lossValue, prediction.getScore(label));
				break;

			case BPA_1NN:
				this.bpa1NnDeletingPolicy(example, exampleAggressiveness, lossValue, prediction.getScore(label)); 
				break;
			}
		}
		return prediction;
	}

	private float evaluateLoss(float prediction, Example example){
		float lossValue = 0;//it represents the distance from the correct semi-space
		if((prediction>0)!=example.isExampleOf(label)){
			lossValue = 1 + Math.abs(prediction);
		}else if(Math.abs(prediction)<1){
			lossValue = 1 - Math.abs(prediction);			
		}		
		return lossValue;
	}

	private void bpaSDeletingPolicy(Example example, float exampleAggressiveness, float loss, float prediction) {

		float exampleSquaredNorm = kernel.squaredNorm(example);
		float minimumObjectiveFuction=exampleAggressiveness*loss;
		float bestNewWeight = 0; //the weight associated to the new support vector when r* is removed 
		//		System.out.println("################################");
		//		System.out.println("prediction before update:" + prediction);
		//		System.out.println("NORM^2: " + this.getPredictionFunction().getModel().getSquaredNorm());
		//		System.out.println("Loss: " + loss);

		int rStarIndex = 0;
		SupportVector rStar = null;

		int svIndex =0;
		for(SupportVector sv : classifier.getModel().getSupportVectors()){

			float newWeight=sv.getWeight()*this.kernel.innerProduct(example, sv.getInstance())/exampleSquaredNorm;
			float tau=loss/exampleSquaredNorm;
			if(tau>exampleAggressiveness){
				tau=exampleAggressiveness;
			}
			if(example.isExampleOf(label)){
				newWeight+=tau;
			}else{
				newWeight-=tau;
			}

			float objectiveFunction = this.evaluateObjectiveFunctionInBpaS(example, newWeight, prediction, sv, exampleAggressiveness);
			//			System.out.println(svIndex + " objectiveFunction: " + objectiveFunction);
			if(objectiveFunction<minimumObjectiveFuction){
				rStarIndex = svIndex;
				rStar = sv;
				bestNewWeight = newWeight;
				minimumObjectiveFuction = objectiveFunction;
			}
			svIndex++;
		}


		if(rStar!=null){

			//if it is convenient I substitute r* (the best SV to be removed) with the new example
			this.getPredictionFunction().getModel().substituteSupportVector(rStarIndex, example, bestNewWeight);
			//			System.out.println("MINIMUM OBJ: " + minimumObjectiveFuction);
			//			System.out.println("r* index: " + rStarIndex);
			//			System.out.println("NEW NORM^2: " + this.getPredictionFunction().getModel().getSquaredNorm());
			//			float pred = this.getPredictionFunction().predict(example).getScore(label);
			//			System.out.println("NEW PREDICTION: " + pred);
			//			System.out.println("LOSS: " + evaluateLoss(pred, example));
		}
	}

	private void bpa1NnDeletingPolicy(Example example, float exampleAggressiveness, float loss, float prediction) {

		float exampleSquaredNorm = kernel.squaredNorm(example);
		float minimumObjectiveFuction=exampleAggressiveness*loss;
		float bestNewWeight = 0; //the weight associated to the new support vector when r* is removed 
		int bestNnIndex = 0;
		float bestNnWeightVariation = 0;
		//		System.out.println("################################");
		//		System.out.println("prediction before update:" + prediction);
		//		System.out.println("NORM^2: " + this.getPredictionFunction().getModel().getSquaredNorm());
		//		System.out.println("Loss: " + loss);

		int rStarIndex = 0;
		SupportVector rStar = null;

		List<SupportVector> svs = this.getPredictionFunction().getModel().getSupportVectors();
		
		int svIndex =0;
		for(SupportVector sv : svs){

			int nn=this.getNearestNeighborIndex(svIndex);

			Example nearestSv = svs.get(nn).getInstance();

			float kNNrT= kernel.innerProduct(nearestSv, example);
			float kNNrNNr= kernel.squaredNorm(nearestSv);
			float ktt= exampleSquaredNorm;

			
			kMatrix.set(0, 0, kNNrNNr);
			kMatrix.set(0,1, kNNrT);
			kMatrix.set(1,0, kNNrT);
			kMatrix.set(1,1, ktt);

			float krNNr = kernel.innerProduct(sv.getInstance(), nearestSv);
			float krt = kernel.innerProduct(sv.getInstance(), example);
			
			krVector.set(0, 0, krNNr);
			krVector.set(1, 0, krt);

			
			ktVector.set(0, 0, kNNrT);
			ktVector.set(1, 0, ktt);

			CommonOps.invert(kMatrix); //now kMatrix is equal to K^-1
			
			CommonOps.mult(kMatrix, krVector, Kkr);

			CommonOps.mult(kMatrix, ktVector, Kkt);

			float yt = -1;
			if(example.isExampleOf(label)){
				yt=1;
			}
			//starting tau computation:
			float Kkrkt = (float) VectorVectorMult.innerProd(Kkr, ktVector); // Kkrkt= (K^-1 * k_r)T * k_t
			float Kktkt = (float) VectorVectorMult.innerProd(Kkt, ktVector); // Kktkt= (K^-1 * k_t)T * k_t

			float tau = (1 - yt * (prediction - sv.getWeight()*krt + sv.getWeight()*Kkrkt))/(Kktkt);  
			if(tau<0){
				tau=0;
			}else if(tau>exampleAggressiveness){
				tau = exampleAggressiveness;
			}

			//starting beta computation:
			
			CommonOps.add(sv.getWeight(), Kkr, tau*yt, Kkt, beta);


			float nnWeightVariation = (float) beta.get(0, 0);

			float newWeight = (float) beta.get(1, 0);

			float objectiveFunction = this.evaluateObjectiveFunctionInBpa1nn(example, newWeight, prediction, sv, nearestSv, nnWeightVariation, exampleAggressiveness);
			//			System.out.println(svIndex + " objectiveFunction: " + objectiveFunction);
			if(objectiveFunction<minimumObjectiveFuction){
				rStarIndex = svIndex;
				rStar = sv;
				bestNewWeight = newWeight;
				minimumObjectiveFuction = objectiveFunction;
				bestNnWeightVariation = nnWeightVariation;
				bestNnIndex = nn;
			}
			svIndex++;
		}


		if(rStar!=null){

			//if it is convenient I substitute r* (the best SV to be removed) with the new example
			this.getPredictionFunction().getModel().substituteSupportVector(rStarIndex, example, bestNewWeight);
			svs.get(bestNnIndex).setWeight(bestNnWeightVariation+svs.get(bestNnIndex).getWeight());//NN Weight modification
			this.updateNearestNeighbors(bestNnIndex);
			//			System.out.println("MINIMUM OBJ: " + minimumObjectiveFuction);
			//			System.out.println("r* index: " + rStarIndex);
			//			System.out.println("NEW NORM^2: " + this.getPredictionFunction().getModel().getSquaredNorm());
			//			float pred = this.getPredictionFunction().predict(example).getScore(label);
			//			System.out.println("NEW PREDICTION: " + pred);
			//			System.out.println("LOSS: " + evaluateLoss(pred, example));
		}
	}


	private float evaluateObjectiveFunctionInBpaS(Example newInstance, float newInstanceWeight, float prediction, SupportVector svToDelete, float exampleAggressiveness) {
		float normVariation = svToDelete.getWeight()*svToDelete.getWeight()*kernel.squaredNorm(svToDelete.getInstance());
		normVariation += newInstanceWeight*newInstanceWeight*kernel.squaredNorm(newInstance);
		normVariation-=2*svToDelete.getWeight()*newInstanceWeight*kernel.innerProduct(newInstance, svToDelete.getInstance());
		float newPrediction = prediction+newInstanceWeight*kernel.squaredNorm(newInstance)-svToDelete.getWeight()*kernel.innerProduct(newInstance, svToDelete.getInstance());
		float loss = evaluateLoss(newPrediction, newInstance);
		//float normVariation = evaluateNormVariation(weightVariation, weightVariationIndex, example, exampleWeight);
		return 0.5f*normVariation+exampleAggressiveness*loss;
	}
	
	private float evaluateObjectiveFunctionInBpa1nn(Example newInstance, float newInstanceWeight, float prediction, SupportVector svToDelete, Example svToModify, float weightVariation, float exampleAggressiveness) {
		float normVariation = svToDelete.getWeight()*svToDelete.getWeight()*kernel.squaredNorm(svToDelete.getInstance());
		normVariation += newInstanceWeight*newInstanceWeight*kernel.squaredNorm(newInstance);
		normVariation += weightVariation*weightVariation*kernel.squaredNorm(svToModify);
		normVariation-=2*svToDelete.getWeight()*newInstanceWeight*kernel.innerProduct(newInstance, svToDelete.getInstance());
		normVariation-=2*svToDelete.getWeight()*weightVariation*kernel.innerProduct(svToDelete.getInstance(), svToModify);
		normVariation+=2*newInstanceWeight*weightVariation*kernel.innerProduct(newInstance, svToModify);
		
		float newPrediction = prediction+newInstanceWeight*kernel.squaredNorm(newInstance)-svToDelete.getWeight()*kernel.innerProduct(newInstance, svToDelete.getInstance()) + weightVariation*kernel.innerProduct(newInstance, svToModify);
		float loss = evaluateLoss(newPrediction, newInstance);
		//float normVariation = evaluateNormVariation(weightVariation, weightVariationIndex, example, exampleWeight);
		return 0.5f*normVariation+exampleAggressiveness*loss;
	}


	@Override
	public void learn(Dataset dataset){
		if(this.fairness){
			float positiveExample = dataset.getNumberOfPositiveExamples(label);
			float negativeExample = dataset.getNumberOfNegativeExamples(label);
			cp = cn * negativeExample / positiveExample;
		}
		//System.out.println("cn: " + c + " cp: " + cp);
		super.learn(dataset);
	}

	@Override
	public BinaryMarginClassifierOutput learn(Example example){

		return (BinaryMarginClassifierOutput) super.learn(example);

	}

	private void computeNearestNeighbors(){
		if(this.nearestNeighbors == null){
			this.nearestNeighbors = new int [budget];
			this.nearestNeighborsSimilarity = new float [budget];
		}
		int svIndex = 0;
		for(SupportVector sv : this.getPredictionFunction().getModel().getSupportVectors()){
			int nn=-1;
			float maxSimilarity=Float.NEGATIVE_INFINITY;
			int svIndex2 = 0;
			for(SupportVector sv2 : this.getPredictionFunction().getModel().getSupportVectors()){
				if(sv!=sv2){
					float currentSimilarity = kernel.innerProduct(sv.getInstance(), sv2.getInstance());
					if(currentSimilarity>maxSimilarity){
						maxSimilarity=currentSimilarity;
						nn=svIndex2;
					}
				}
				svIndex2++;
			}

			this.nearestNeighbors[svIndex]=nn;
			this.nearestNeighborsSimilarity[svIndex]=maxSimilarity;
		}
		this.areNnComputed=true;

	}

	private int getNearestNeighborIndex(int svIndex){
		if(!this.areNnComputed){
			this.computeNearestNeighbors();
		}
		return this.nearestNeighbors[svIndex];
	}

	private void updateNearestNeighbors(int changedSvIndex){

		List<SupportVector> svs = this.getPredictionFunction().getModel().getSupportVectors();

		Example newSv = svs.get(changedSvIndex).getInstance();
		float currentSimilarity;
		int changedSvNN=-1;
		float maxSimilarityChangedSv=Float.NEGATIVE_INFINITY;
		for(int i=0; i<this.budget; i++){
			if(i==changedSvIndex){
				continue;
			}

			Example currentExample = svs.get(i).getInstance();
			currentSimilarity = kernel.innerProduct(newSv, currentExample);
			if(currentSimilarity>maxSimilarityChangedSv){
				maxSimilarityChangedSv=currentSimilarity;
				changedSvNN=i;
			}
			if(this.nearestNeighbors[i]==changedSvIndex){//the nearest neighbor of the current one has been removed
				int nn2=-1;
				float maxSimilarity2=Float.NEGATIVE_INFINITY;
				for(int j=0; j<this.budget; j++){
					if(j==i){
						continue;
					}
					float currentSimilarity2 = kernel.innerProduct(svs.get(j).getInstance(), currentExample);
					if(currentSimilarity2>maxSimilarity2){
						maxSimilarity2=currentSimilarity2;
						nn2=j;
					}
				}
				this.nearestNeighbors[i] = nn2;
				this.nearestNeighborsSimilarity[i] = maxSimilarity2;
			}
			else if(currentSimilarity>this.nearestNeighborsSimilarity[i]){//checking whether the new Sv is closer to the old NN
				this.nearestNeighborsSimilarity[i]=currentSimilarity;
				this.nearestNeighbors[i] = changedSvIndex;
			}


		}

		this.nearestNeighbors[changedSvIndex] = changedSvNN;
		this.nearestNeighborsSimilarity[changedSvIndex] = maxSimilarityChangedSv;


	}


}
