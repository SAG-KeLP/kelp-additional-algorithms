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

package it.uniroma2.sag.kelp.learningalgorithm.clustering.kernelbasedkmeans;

import it.uniroma2.sag.kelp.data.clustering.Cluster;
import it.uniroma2.sag.kelp.data.clustering.ClusterExample;
import it.uniroma2.sag.kelp.data.clustering.ClusterList;
import it.uniroma2.sag.kelp.data.dataset.Dataset;
import it.uniroma2.sag.kelp.data.dataset.selector.ExampleSelector;
import it.uniroma2.sag.kelp.data.dataset.selector.FirstExamplesSelector;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.kernel.Kernel;
import it.uniroma2.sag.kelp.learningalgorithm.clustering.ClusteringAlgorithm;

import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonTypeName;

/**
 * Implements the Kernel Based K-means described in:
 * 
 * Brian Kulis, Sugato Basu, Inderjit Dhillon, and Raymond Mooney.
 * Semi-supervised graph clustering: a kernel approach. Machine Learning,
 * 74(1):1-22, January 2009.
 * 
 * 
 * @author Danilo Croce
 */
@JsonTypeName("kernelbased_kmeans")
public class KernelBasedKMeansEngine implements ClusteringAlgorithm {
	private Logger logger = LoggerFactory
			.getLogger(KernelBasedKMeansEngine.class);

	/**
	 * The Kernel Function
	 */
	private Kernel kernel;

	/**
	 * The number of expected clusters
	 */
	private int k;

	/**
	 * The maximum number of iterations
	 */
	private int maxIterations;

	/**
	 * Stores the example Weight. These weight are still not used in this
	 * release
	 */
	@JsonIgnore
	private HashMap<Example, Float> alphas;

	/**
	 * Stores part of the computation of the distance function
	 */
	@JsonIgnore
	private HashMap<Cluster, Float> thirdMemberEqBuffer;

	public KernelBasedKMeansEngine() {
		super();
		alphas = new HashMap<Example, Float>();
		thirdMemberEqBuffer = new HashMap<Cluster, Float>();
	}

	/**
	 * @param kernel
	 *            The kernel function
	 * @param k
	 *            The number of expected clusters
	 * @param maxIterations
	 *            The maximum number of iterations
	 */
	public KernelBasedKMeansEngine(Kernel kernel, int k, int maxIterations) {
		this();
		this.kernel = kernel;
		this.k = k;
		this.maxIterations = maxIterations;
	}

	/**
	 * Estimate the distance of an example from the centroid
	 * 
	 * @param example
	 *            An example
	 * @param cluster
	 *            A cluster
	 * @return The distance
	 */
	public float calculateDistance(Example example, Cluster cluster) {

		// res+=k_ii
		float first = evaluateKernel(example, example);

		// 2 * ( sum_{x{j}\in \PI_{c}} \alpha_{j} Kij )
		float secondNum = 0;
		float secondDen = 0;

		for (int j = 0; j < cluster.size(); j++) {
			Example e_j = cluster.getExamples().get(j).getExample();
			secondNum += getAlpha(e_j)
					* evaluateKernel(example, cluster.getExamples().get(j)
							.getExample());
			secondDen += getAlpha(e_j);
		}
		secondNum *= 2.0;

		float thirdMember = 0;

		// thir member
		if (thirdMemberEqBuffer.get(cluster) == null) {
			float thirdNum = 0;
			float thirdDen = 0;

			for (int j = 0; j < cluster.size(); j++) {
				Example e_j = cluster.getExamples().get(j).getExample();
				for (int l = 0; l < cluster.size(); l++) {
					Example e_l = cluster.getExamples().get(l).getExample();

					thirdNum += getAlpha(e_j) * getAlpha(e_l)
							* evaluateKernel(e_j, e_l);
				}
				thirdDen += getAlpha(e_j);
			}

			thirdDen *= thirdDen;

			thirdMemberEqBuffer.put(cluster, thirdNum / thirdDen);
		}

		thirdMember = thirdMemberEqBuffer.get(cluster);

		return (float) Math.sqrt(first - secondNum / secondDen + thirdMember);
	}

	public void checkConsistency(int K, int inputSize) throws Exception {
		if (inputSize < K) {
			throw new Exception("Error: the number of instances (" + inputSize
					+ ") must be higher than k (" + K + ")");
		}
	}

	@Override
	public ClusterList cluster(Dataset dataset) {
		return cluster(dataset, new FirstExamplesSelector(k));
	}

	@Override
	public ClusterList cluster(Dataset dataset, ExampleSelector seedSelector) {
		/*
		 * Check consistency: the number of input examples MUST be greater or
		 * equal to the target K
		 */
		if (dataset.getNumberOfExamples() < k) {
			System.err.println("Error: the number of instances ("
					+ dataset.getNumberOfExamples()
					+ ") must be higher than k (" + k + ")");
			return null;
		}

		/*
		 * Alphas Value are stored
		 */
		for (Example example : dataset.getExamples()) {
			alphas.put(example, 1.0f);
		}

		/*
		 * Initialize seed and outputStructures
		 */
		ClusterList resClusters = new ClusterList();
		List<Example> seedVector = seedSelector.select(dataset);
		for (int clusterId = 0; clusterId < k; clusterId++) {
			resClusters.add(new Cluster("cluster_" + clusterId));
			if (clusterId < seedVector.size()) {
				KernelBasedKMeansExample kernelBasedKMeansExample = new KernelBasedKMeansExample(
						seedVector.get(clusterId), 0);

				resClusters.get(clusterId).add(kernelBasedKMeansExample);
			}
		}

		/*
		 * Do Work
		 */
		// For each iteration
		for (int t = 0; t < maxIterations; t++) {

			int reassignment;

			logger.debug("\nITERATION:\t" + (t + 1));

			TreeMap<Long, Integer> exampleIdToClusterMap = new TreeMap<Long, Integer>();

			HashMap<Example, Float> minDistances = new HashMap<Example, Float>();

			/*
			 * Searching for the nearest cluster
			 */
			for (Example example : dataset.getExamples()) {

				float minValue = Float.MAX_VALUE;
				int targetCluster = -1;

				for (int clusterId = 0; clusterId < k; clusterId++) {

					float d = calculateDistance(example,
							resClusters.get(clusterId));

					logger.debug("Distance of " + example.getId()
							+ " from cluster " + clusterId + ":\t" + d);

					if (d < minValue) {
						minValue = d;
						targetCluster = clusterId;
					}
				}

				minDistances.put(example, minValue);
				exampleIdToClusterMap.put(example.getId(), targetCluster);
			}

			/*
			 * Counting reassignments
			 */
			reassignment = countReassigment(exampleIdToClusterMap, resClusters);

			logger.debug("Reassigments:\t" + reassignment);

			/*
			 * Updating
			 */
			for (int i = 0; i < resClusters.size(); i++)
				resClusters.get(i).clear();
			this.thirdMemberEqBuffer.clear();

			for (Example example : dataset.getExamples()) {
				logger.debug("Re-assigning " + example.getId() + " to "
						+ exampleIdToClusterMap.get(example.getId()));

				int assignedClusterId = exampleIdToClusterMap.get(example
						.getId());
				float minDist = minDistances.get(example);

				KernelBasedKMeansExample kernelBasedKMeansExample = new KernelBasedKMeansExample(
						example, minDist);

				resClusters.get(assignedClusterId)
						.add(kernelBasedKMeansExample);
			}

			if (t > 0 && reassignment == 0) {
				break;
			}
		}

		/*
		 * Sort results by distance from the controid.
		 */
		for (Cluster c : resClusters) {
			c.sortAscendingOrder();
		}

		return resClusters;

	}

	/**
	 * Count the reassignment as a stopping criteria for the algorithm
	 * 
	 * @param exampleIdToClusterMap
	 *            The map of assignment for the previous iteration
	 * @param clusterList
	 *            The actual clusters
	 * @return
	 */
	private int countReassigment(TreeMap<Long, Integer> exampleIdToClusterMap,
			List<Cluster> clusterList) {

		int reassignment = 0;

		TreeMap<Long, Integer> currentExampleIdToClusterMap = new TreeMap<Long, Integer>();

		int clusterId = 0;
		for (Cluster cluster : clusterList) {
			for (ClusterExample clusterExample : cluster.getExamples()) {
				currentExampleIdToClusterMap.put(clusterExample.getExample()
						.getId(), clusterId);
			}
			clusterId++;
		}

		for (Long currentExId : currentExampleIdToClusterMap.keySet()) {
			if (exampleIdToClusterMap.get(currentExId).intValue() != currentExampleIdToClusterMap
					.get(currentExId).intValue())
				reassignment++;
		}

		return reassignment;
	}

	public float evaluateKernel(Example e1, Example e2) {
		return kernel.innerProduct(e1, e2);
	}

	@JsonIgnore
	private float getAlpha(Example example) {
		return alphas.get(example);
	}

	public int getK() {
		return k;
	}

	public Kernel getKernel() {
		return kernel;
	}

	public int getMaxIterations() {
		return maxIterations;
	}

	public void setK(int k) {
		this.k = k;
	}

	public void setKernel(Kernel kernel) {
		this.kernel = kernel;
	}

	public void setMaxIterations(int maxIterations) {
		this.maxIterations = maxIterations;
	}

}
