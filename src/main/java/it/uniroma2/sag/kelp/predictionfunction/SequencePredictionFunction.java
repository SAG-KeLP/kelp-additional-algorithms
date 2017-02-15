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

package it.uniroma2.sag.kelp.predictionfunction;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Vector;

import org.apache.commons.lang3.SerializationUtils;

import com.fasterxml.jackson.annotation.JsonIgnore;

import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.SequenceExample;
import it.uniroma2.sag.kelp.data.example.SequencePath;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.label.SequenceEmission;
import it.uniroma2.sag.kelp.predictionfunction.model.Model;
import it.uniroma2.sag.kelp.predictionfunction.model.SequenceModel;

/**
 * 
 * This class implements a classifier in a sequence labeling process. During
 * this labeling process, the history of an example is dynamically estimated by
 * a classifier and the entire sequence of labels is derived through a Viterbi
 * Decoding step combined with a Beam Search Strategy.
 * 
 * @author Danilo Croce
 *
 */
public class SequencePredictionFunction implements PredictionFunction {


	public static final int DEFAULT_MAX_EMISSION_CAND = 5;

	public static final int DEFAULT_BEAM_SIZE = 20;

	/**
	 * During the labeling process, each item is classified with respect to the
	 * target classes. To reduce the complexity of the labeling process, this
	 * variable determines the number of classes that received the highest
	 * classification scores to be considered after the classification step in
	 * the Viterbi Decoding.
	 */
	private int maxEmissionCandidates;

	/**
	 * The size of the beam to be used in the decoding process. This number
	 * determines the number of possible sequences produced in the labeling
	 * process. It will also increase the process complexity.
	 */
	private int beamSize;

	/**
	 * The model produced by a SequenceClassificationLearningAlgorithm
	 */
	private SequenceModel model;

	public SequencePredictionFunction() {
		super();
		this.maxEmissionCandidates = DEFAULT_MAX_EMISSION_CAND;
		this.beamSize = DEFAULT_BEAM_SIZE;
	}

	/**
	 * @param model
	 *            The model produced by a
	 *            <code>SequenceClassificationLearningAlgorithm</code>
	 */
	public SequencePredictionFunction(SequenceModel model) {
		this();
		this.model = model;
	}

	/**
	 * @return The size of the beam to be used in the decoding process. This
	 *         number determines the number of possible sequences produced in
	 *         the labeling process. It will also increase the process
	 *         complexity.
	 */
	public int getBeamSize() {
		return beamSize;
	}

	/**
	 * Given the classification scores produced by the instance based
	 * classification function, these are converted into a sort of probability
	 * by applying the softmax operator
	 * 
	 * @param observationsPrediction
	 * @return
	 */
	private HashMap<Label, Float> getEmissionsProbabilities(Prediction observationsPrediction) {
		double sum = 0;
		List<Label> labels = model.getBasePredictionFunction().getLabels();
		for (Label label : labels) {
			sum += Math.exp(observationsPrediction.getScore(label));
		}

		HashMap<Label, Float> emissionProbabilities = new HashMap<Label, Float>();
		for (Label label : labels) {
			emissionProbabilities.put(label, (float) (Math.exp(observationsPrediction.getScore(label)) / sum));
		}
		return emissionProbabilities;
	}

	@JsonIgnore
	@Override
	public List<Label> getLabels() {
		return model.getBasePredictionFunction().getLabels();
	}

	/**
	 * @return During the labeling process, each item is classified with respect
	 *         to the target classes. To reduce the complexity of the labeling
	 *         process, the returned variable determines the number of classes
	 *         that received the highest classification scores to be considered
	 *         after the classification step in the Viterbi Decoding.
	 */
	public int getMaxEmissionCandidates() {
		return maxEmissionCandidates;
	}

	@Override
	public SequenceModel getModel() {
		return model;
	}

	@Override
	public SequencePrediction predict(Example example) {

		SequenceExample sequenceExample = (SequenceExample) example;

		int sequenceLenght = sequenceExample.getLenght();

		List<SequencePath> resBeam = new ArrayList<SequencePath>();

		/*
		 * Intial states
		 */
		SequencePath initaliaState = new SequencePath();
		resBeam.add(initaliaState);

		for (int elementId = 0; elementId < sequenceLenght; elementId++) {
			List<SequencePath> newBeam = new ArrayList<SequencePath>();

			/*
			 * For each path from the beam
			 */
			for (SequencePath p : resBeam) {
				Prediction observationsPrediction = null;

				/*
				 * Get the previous classifications
				 */
				Example enrichedObservedExample = model.getSequenceExampleGenerator()
						.generateExampleWithHistory(sequenceExample, p, elementId);

				/*
				 * Classify the observation
				 */
				observationsPrediction = model.getBasePredictionFunction().predict(enrichedObservedExample);

				/*
				 * Get the emission probability
				 */
				HashMap<Label, Float> emissionProbabilities = getEmissionsProbabilities(observationsPrediction);

				/*
				 * Order the states by the emission probability
				 */
				List<SequenceEmission> orderedStates = new Vector<SequenceEmission>();
				for (Label label : getLabels()) {
					orderedStates.add(new SequenceEmission(label, emissionProbabilities.get(label)));
				}
				Collections.sort(orderedStates);
				Collections.reverse(orderedStates);

				/*
				 * If <code>transitionOrder==0</code> adding only the emission
				 * probability and jumping to the next example
				 */
				if (model.getSequenceExampleGenerator().getTransitionsOrder() == 0) {
					SequenceEmission bestState = orderedStates.get(0);
					SequenceEmission sequenceLabel = new SequenceEmission(bestState.getLabel(), bestState.getEmission());
					p.add(sequenceLabel);
					p.setScore(initaliaState.getScore() + Math.log(bestState.getEmission()));
					newBeam.add(p);
					continue;
				}

				/*
				 * Selecting the $maxEmissionCandidates$ most probable classes
				 */
				if (orderedStates.size() >= maxEmissionCandidates)
					orderedStates = orderedStates.subList(0, maxEmissionCandidates);

				for (SequenceEmission scoredLabel : orderedStates) {

					Label label = scoredLabel.getLabel();

					SequencePath newPath = SerializationUtils.clone(p);

					float emissionProbability = emissionProbabilities.get(label);

					/*
					 * Adding a new state in the path that is the sum of the
					 * emission and prior probability
					 */
					SequenceEmission newSequenceLabel = new SequenceEmission(label, emissionProbability);
					newPath.getAssignedSequnceLabels().add(newSequenceLabel);
					newPath.setScore(newPath.getScore() + Math.log(emissionProbability));

					/*
					 * Adding the path to the beam
					 */
					newBeam.add(newPath);
				}
			}

			/*
			 * Selecting the $beamWidth$ most probable paths from the beam
			 */
			Collections.sort(newBeam);
			Collections.reverse(newBeam);

			if (newBeam.size() > beamSize) {
				for (int i = newBeam.size() - 1; i >= beamSize; i--) {
					newBeam.remove(i);
				}
			}

			resBeam = newBeam;
		}

		/*
		 * Returing the prediction with the $beamWidth$ most probable paths
		 */
		SequencePrediction prediction = new SequencePrediction();
		prediction.setPaths(resBeam);
		return prediction;
	}

	@Override
	public void reset() {
		this.model.getBasePredictionFunction().reset();
		this.model.reset();
	}

	/**
	 * @param beamSize
	 *            The size of the beam to be used in the decoding process. This
	 *            number determines the number of possible sequences produced in
	 *            the labeling process. It will also increase the process
	 *            complexity.
	 */
	public void setBeamSize(int beamSize) {
		this.beamSize = beamSize;
	}

	@Override
	@JsonIgnore
	public void setLabels(List<Label> labels) {
		this.model.getBasePredictionFunction().setLabels(labels);
	}

	/**
	 * @param maxEmissionCandidates
	 *            During the labeling process, each item is classified with
	 *            respect to the target classes. To reduce the complexity of the
	 *            labeling process, this variable determines the number of
	 *            classes that received the highest classification scores to be
	 *            considered after the classification step in the Viterbi
	 *            Decoding.
	 */
	public void setMaxEmissionCandidates(int maxEmissionCandidates) {
		this.maxEmissionCandidates = maxEmissionCandidates;
	}

	@Override
	public void setModel(Model model) {
		this.model = (SequenceModel) model;
	}

}
