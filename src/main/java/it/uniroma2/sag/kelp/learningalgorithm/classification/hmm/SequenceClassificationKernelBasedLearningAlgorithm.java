/*
 * Copyright 2016 Simone Filice and Giuseppe Castellucci and Danilo Croce and Roberto Basili
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

package it.uniroma2.sag.kelp.learningalgorithm.classification.hmm;

import it.uniroma2.sag.kelp.data.examplegenerator.SequenceExampleGenerator;
import it.uniroma2.sag.kelp.data.examplegenerator.SequenceExampleGeneratorKernel;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.kernel.cache.KernelCache;
import it.uniroma2.sag.kelp.kernel.standard.LinearKernelCombination;
import it.uniroma2.sag.kelp.kernel.vector.LinearKernel;
import it.uniroma2.sag.kelp.learningalgorithm.BinaryLearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.KernelMethod;
import it.uniroma2.sag.kelp.learningalgorithm.LearningAlgorithm;
import it.uniroma2.sag.kelp.learningalgorithm.classification.multiclassification.OneVsAllLearning;

/**
 * /** This class implements a sequential labeling paradigm. <br>
 * Given sequences of items (each implemented as an <code>Example</code> and
 * associated to one <code>Label</code>) this class allow to apply a generic
 * <code>LearningAlgorithm</code> to use the "history" of each item in the
 * sequence in order to improve the classification quality. In other words, the
 * classification of each example does not depend only its representation, but
 * it also depend on its "history", in terms of the classed assigned to the
 * preceding examples. <br>
 * This class should be used when a <b>kernel-based learning algorithm</b> is
 * used, thus directly operating in the implicit space underlying a kernel
 * function.
 * 
 * <br>
 * This algorithms was inspired by the work of: <br>
 * Y. Altun, I. Tsochantaridis, and T. Hofmann. Hidden Markov support vector
 * machines. In Proceedings of the Twentieth International Conference on Machine
 * Learning, 2003.
 * 
 * @author Danilo Croce
 *
 */
public class SequenceClassificationKernelBasedLearningAlgorithm extends SequenceClassificationLearningAlgorithm
		implements KernelMethod {

	private final static String TRANSITION_REPRESENTATION_NAME = "__trans_rep__";

	private LinearKernelCombination sequenceBasedKernel;

	public SequenceClassificationKernelBasedLearningAlgorithm() {

	}

	/**
	 * @param baseLearningAlgorithm
	 *            the learning algorithm devoted to the acquisition of a model
	 *            after that each example has been enriched with its "history"
	 * @param transitionsOrder
	 *            given a targeted item in the sequence, this variable
	 *            determines the number of previous example considered in the
	 *            learning/labeling process.
	 * @param transitionWeight
	 *            the importance of the transition-based features during the
	 *            learning process. Higher valuers will assign more importance
	 *            to the transitions.
	 * @throws Exception
	 *             The input <code>baseLearningAlgorithm</code> is not a
	 *             kernel-based method
	 */
	public SequenceClassificationKernelBasedLearningAlgorithm(BinaryLearningAlgorithm baseLearningAlgorithm,
			int transitionsOrder, float transitionWeight) throws Exception {

		if (!(baseLearningAlgorithm instanceof KernelMethod)) {
			throw new Exception("ERROR: the input baseLearningAlgorithm is not a kernel-based method!");
		}

		Kernel inputKernel = ((KernelMethod) baseLearningAlgorithm).getKernel();

		sequenceBasedKernel = new LinearKernelCombination();
		sequenceBasedKernel.addKernel(1, inputKernel);
		Kernel transitionBasedKernel = new LinearKernel(TRANSITION_REPRESENTATION_NAME);
		sequenceBasedKernel.addKernel(transitionWeight, transitionBasedKernel);
		sequenceBasedKernel.normalizeWeights();

		setKernel(sequenceBasedKernel);

		BinaryLearningAlgorithm binaryLearningAlgorithmCopy = (BinaryLearningAlgorithm) baseLearningAlgorithm
				.duplicate();

		((KernelMethod) binaryLearningAlgorithmCopy).setKernel(sequenceBasedKernel);

		OneVsAllLearning oneVsAllLearning = new OneVsAllLearning();
		oneVsAllLearning.setBaseAlgorithm(binaryLearningAlgorithmCopy);

		super.setBaseLearningAlgorithm(oneVsAllLearning);

		SequenceExampleGenerator sequenceExamplesGenerator = new SequenceExampleGeneratorKernel(
				transitionsOrder, TRANSITION_REPRESENTATION_NAME);

		super.setSequenceExampleGenerator(sequenceExamplesGenerator);
	}

	@Override
	public LearningAlgorithm duplicate() {
		return null;
	}

	@Override
	public LearningAlgorithm getBaseAlgorithm() {
		return super.getBaseLearningAlgorithm();
	}

	@Override
	public Kernel getKernel() {
		return sequenceBasedKernel;
	}

	@Override
	public void setBaseAlgorithm(LearningAlgorithm baseAlgorithm) {
		super.setBaseLearningAlgorithm(baseAlgorithm);
	}

	@Override
	public void setKernel(Kernel kernel) {
		this.sequenceBasedKernel = (LinearKernelCombination) kernel;
	}

	public void setKernelCache(KernelCache cache) {
		this.getKernel().setKernelCache(cache);
	}

}
