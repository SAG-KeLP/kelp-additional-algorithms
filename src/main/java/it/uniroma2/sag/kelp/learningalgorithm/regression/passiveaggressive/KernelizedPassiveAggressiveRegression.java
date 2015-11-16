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
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.learningalgorithm.KernelMethod;
import it.uniroma2.sag.kelp.predictionfunction.regressionfunction.UnivariateKernelMachineRegressionFunction;

/**
 * Online Passive-Aggressive Learning Algorithm for regression tasks (kernel machine version).
 *
 * reference: 
 * <p>
 * [CrammerJLMR2006] Koby Crammer, Ofer Dekel, Joseph Keshet, Shai Shalev-Shwartz and Yoram Singer
 * Online Passive-Aggressive Algorithms. Journal of Machine Learning Research (2006)
 * 
 * @author      Simone Filice
 */
@JsonTypeName("kernelizedPA-R")
public class KernelizedPassiveAggressiveRegression extends PassiveAggressiveRegression implements KernelMethod{

	private Kernel kernel;
	
	public KernelizedPassiveAggressiveRegression(){
		this.regressor = new UnivariateKernelMachineRegressionFunction();
	}
	
	public KernelizedPassiveAggressiveRegression(float aggressiveness, float epsilon, Policy policy, Kernel kernel, Label label){
		this.regressor = new UnivariateKernelMachineRegressionFunction();
		this.setC(aggressiveness);
		this.setEpsilon(epsilon);
		this.setPolicy(policy);
		this.setKernel(kernel);
		this.setLabel(label);
	}
	
	@Override
	public Kernel getKernel(){
		return kernel;
	}

	@Override
	public void setKernel(Kernel kernel) {
		this.kernel = kernel;
		this.getPredictionFunction().getModel().setKernel(kernel);
	}
	
	@Override
	public KernelizedPassiveAggressiveRegression duplicate() {
		KernelizedPassiveAggressiveRegression copy = new KernelizedPassiveAggressiveRegression();
		copy.setC(this.c);
		copy.setKernel(this.kernel);
		copy.setPolicy(this.policy);
		copy.setEpsilon(epsilon);
		return copy;
	}
	
	@Override
	public UnivariateKernelMachineRegressionFunction getPredictionFunction(){
		return (UnivariateKernelMachineRegressionFunction) this.regressor;
	}

}
