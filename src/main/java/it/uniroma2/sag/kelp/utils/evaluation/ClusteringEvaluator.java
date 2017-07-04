package it.uniroma2.sag.kelp.utils.evaluation;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.TreeMap;

import it.uniroma2.sag.kelp.data.clustering.Cluster;
import it.uniroma2.sag.kelp.data.clustering.ClusterExample;
import it.uniroma2.sag.kelp.data.clustering.ClusterList;
import it.uniroma2.sag.kelp.data.example.Example;
import it.uniroma2.sag.kelp.data.example.SimpleExample;
import it.uniroma2.sag.kelp.data.label.Label;
import it.uniroma2.sag.kelp.data.label.StringLabel;
import it.uniroma2.sag.kelp.learningalgorithm.clustering.kernelbasedkmeans.KernelBasedKMeansExample;

/**
 * 
 * Implements Evaluation methods for clustering algorithms.
 * 
 * More details about Purity and NMI can be found here:<br>
 * <br>
 * https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.
 * html
 * 
 * @author Danilo Croce
 *
 */
public class ClusteringEvaluator {

	public static float getPurity(ClusterList clusters) {

		float res = 0;
		int k = clusters.size();

		for (int clustId = 0; clustId < k; clustId++) {

			TreeMap<Label, Integer> classSizes = new TreeMap<Label, Integer>();

			for (ClusterExample vce : clusters.get(clustId).getExamples()) {
				HashSet<Label> labels = vce.getExample().getClassificationLabels();
				for (Label label : labels)
					if (!classSizes.containsKey(label))
						classSizes.put(label, 1);
					else
						classSizes.put(label, classSizes.get(label) + 1);
			}

			int maxSize = 0;
			for (int size : classSizes.values()) {
				if (size > maxSize) {
					maxSize = size;
				}
			}
			res += maxSize;
		}

		return res / (float) clusters.getNumberOfExamples();
	}

	public static float getMI(ClusterList clusters) {

		float res = 0;

		float N = clusters.getNumberOfExamples();

		int k = clusters.size();

		TreeMap<Label, Integer> classCardinality = getClassCardinality(clusters);

		for (int clustId = 0; clustId < k; clustId++) {

			TreeMap<Label, Integer> classSizes = getClassCardinalityWithinCluster(clusters, clustId);

			for (Label className : classSizes.keySet()) {
				int wSize = classSizes.get(className);
				res += ((float) wSize / N) * myLog(N * (float) wSize
						/ (clusters.get(clustId).getExamples().size() * (float) classCardinality.get(className)));
			}

		}

		return res;

	}

	private static TreeMap<Label, Integer> getClassCardinalityWithinCluster(ClusterList clusters, int clustId) {

		TreeMap<Label, Integer> classSizes = new TreeMap<Label, Integer>();

		for (ClusterExample vce : clusters.get(clustId).getExamples()) {
			HashSet<Label> labels = vce.getExample().getClassificationLabels();
			for (Label label : labels)
				if (!classSizes.containsKey(label))
					classSizes.put(label, 1);
				else
					classSizes.put(label, classSizes.get(label) + 1);
		}

		return classSizes;
	}

	private static float getClusterEntropy(ClusterList clusters) {

		float res = 0;
		float N = clusters.getNumberOfExamples();
		int k = clusters.size();

		for (int clustId = 0; clustId < k; clustId++) {
			int clusterElementSize = clusters.get(clustId).getExamples().size();
			if (clusterElementSize != 0)
				res -= ((float) clusterElementSize / N) * myLog((float) clusterElementSize / N);
		}
		return res;

	}

	private static float getClassEntropy(ClusterList clusters) {

		float res = 0;
		float N = clusters.getNumberOfExamples();

		TreeMap<Label, Integer> classCardinality = getClassCardinality(clusters);

		for (int classSize : classCardinality.values()) {
			res -= ((float) classSize / N) * myLog((float) classSize / N);
		}
		return res;

	}

	private static float myLog(float f) {
		return (float) (Math.log(f) / Math.log(2f));
	}

	private static TreeMap<Label, Integer> getClassCardinality(ClusterList clusters) {
		TreeMap<Label, Integer> classSizes = new TreeMap<Label, Integer>();

		int k = clusters.size();

		for (int clustId = 0; clustId < k; clustId++) {

			for (ClusterExample vce : clusters.get(clustId).getExamples()) {
				HashSet<Label> labels = vce.getExample().getClassificationLabels();
				for (Label label : labels)
					if (!classSizes.containsKey(label))
						classSizes.put(label, 1);
					else
						classSizes.put(label, classSizes.get(label) + 1);
			}
		}
		return classSizes;
	}

	public static float getNMI(ClusterList clusters) {
		return getMI(clusters) / ((getClusterEntropy(clusters) + getClassEntropy(clusters)) / 2f);
	}

	public static String getStatistics(ClusterList clusters) {
		StringBuilder sb = new StringBuilder();

		sb.append("Purity:\t" + getPurity(clusters) + "\n");
		sb.append("Mutual Information:\t" + getMI(clusters) + "\n");
		sb.append("Cluster Entropy:\t" + getClusterEntropy(clusters) + "\n");
		sb.append("Class Entropy:\t" + getClassEntropy(clusters) + "\n");
		sb.append("NMI:\t" + getNMI(clusters));

		return sb.toString();
	}

	public static void main(String[] args) {
		ClusterList clusters = new ClusterList();

		Cluster c1 = new Cluster("C1");
		ArrayList<Example> list1 = new ArrayList<Example>();
		list1.add(new SimpleExample(new StringLabel[] { new StringLabel("x") }, null));
		list1.add(new SimpleExample(new StringLabel[] { new StringLabel("x") }, null));
		list1.add(new SimpleExample(new StringLabel[] { new StringLabel("x") }, null));
		list1.add(new SimpleExample(new StringLabel[] { new StringLabel("x") }, null));
		list1.add(new SimpleExample(new StringLabel[] { new StringLabel("x") }, null));
		list1.add(new SimpleExample(new StringLabel[] { new StringLabel("o") }, null));
		for (Example e : list1) {
			c1.add(new KernelBasedKMeansExample(e, 1f));
		}

		Cluster c2 = new Cluster("C2");
		ArrayList<Example> list2 = new ArrayList<Example>();
		list2.add(new SimpleExample(new StringLabel[] { new StringLabel("x") }, null));
		list2.add(new SimpleExample(new StringLabel[] { new StringLabel("o") }, null));
		list2.add(new SimpleExample(new StringLabel[] { new StringLabel("o") }, null));
		list2.add(new SimpleExample(new StringLabel[] { new StringLabel("o") }, null));
		list2.add(new SimpleExample(new StringLabel[] { new StringLabel("o") }, null));
		list2.add(new SimpleExample(new StringLabel[] { new StringLabel("q") }, null));
		for (Example e : list2) {
			c2.add(new KernelBasedKMeansExample(e, 1f));
		}

		Cluster c3 = new Cluster("C3");
		ArrayList<Example> list3 = new ArrayList<Example>();
		list3.add(new SimpleExample(new StringLabel[] { new StringLabel("q") }, null));
		list3.add(new SimpleExample(new StringLabel[] { new StringLabel("q") }, null));
		list3.add(new SimpleExample(new StringLabel[] { new StringLabel("q") }, null));
		list3.add(new SimpleExample(new StringLabel[] { new StringLabel("x") }, null));
		list3.add(new SimpleExample(new StringLabel[] { new StringLabel("x") }, null));
		for (Example e : list3) {
			c3.add(new KernelBasedKMeansExample(e, 1f));
		}
		
		clusters.add(c1);
		clusters.add(c2);
		clusters.add(c3);
		
		System.out.println(ClusteringEvaluator.getStatistics(clusters));
		
		//From https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
		//Purity = 0.71
		//NMI = 0.36
		
	}

}
