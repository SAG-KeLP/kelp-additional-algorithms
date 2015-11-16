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

package it.uniroma2.sag.kelp.learningalgorithm.classification.liblinear.solver;

/**
 * NOTE: This code has been adapted from the Java port of the original LIBLINEAR
 * C++ sources. Original Java sources (v 1.94) are available at
 * <code>http://liblinear.bwaldvogel.de</code>
 * 
 * @author Danilo Croce
 */
public class LibLinearFeatureNode implements LibLinearFeature {

    public final int index;
    public double    value;

    public LibLinearFeatureNode( final int index, final double value ) {
        if (index < 0) throw new IllegalArgumentException("index must be >= 0");
        this.index = index;
        this.value = value;
    }

    /**
     * @since 1.9
     */
    public int getIndex() {
        return index;
    }

    /**
     * @since 1.9
     */
    public double getValue() {
        return value;
    }

    /**
     * @since 1.9
     */
    public void setValue(double value) {
        this.value = value;
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + index;
        long temp;
        temp = Double.doubleToLongBits(value);
        result = prime * result + (int)(temp ^ (temp >>> 32));
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null) return false;
        if (getClass() != obj.getClass()) return false;
        LibLinearFeatureNode other = (LibLinearFeatureNode)obj;
        if (index != other.index) return false;
        if (Double.doubleToLongBits(value) != Double.doubleToLongBits(other.value)) return false;
        return true;
    }

    @Override
    public String toString() {
        return "FeatureNode(idx=" + index + ", value=" + value + ")";
    }
}
