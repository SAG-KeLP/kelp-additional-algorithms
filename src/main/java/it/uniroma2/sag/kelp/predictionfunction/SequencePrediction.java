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
import java.util.List;

import it.uniroma2.sag.kelp.data.example.SequencePath;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.predictionfunction.Prediction;

/**
 * It is a output provided by a machine learning systems on a sequence. This
 * specific implementation allows to assign multiple labelings to single
 * sequence, useful for some labeling strategies, such as Beam Search. Notice
 * that each labeling requires a score to select the more promising labeling.
 * 
 * @author Danilo Croce
 *
 */
public class SequencePrediction implements Prediction {

	/**
	 * This list contains multiple labelings to be assigned to a single sequence
	 */
	private List<SequencePath> paths;

	public SequencePrediction() {
		paths = new ArrayList<SequencePath>();
	}

	/**
	 * @return The best path, i.e., the labeling with the highest score in the
	 *         list of labelings provided by a classifier
	 */
	public SequencePath bestPath() {
		return paths.get(0);
	}

	/**
	 * @return a list containing multiple labelings to be assigned to a single
	 *         sequence
	 */
	public List<SequencePath> getPaths() {
		return paths;
	}

	@Override
	public Float getScore(Label label) {
		return null;
	}

	/**
	 * @param paths
	 *            a list contains multiple labelings to be assigned to a single
	 *            sequence
	 */
	public void setPaths(List<SequencePath> paths) {
		this.paths = paths;
	}

	@Override
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < paths.size(); i++) {
			if (i == 0)
				sb.append("Best Path\t");
			else
				sb.append("Altern. Path\t");
			SequencePath sequencePath = paths.get(i);
			sb.append(sequencePath + "\n");
		}
		return sb.toString();
	}

}
