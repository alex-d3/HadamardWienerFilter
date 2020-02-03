using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;

using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Random;
using MathNet.Numerics.Distributions;

namespace HadamardWienerFilter
{
    internal class TransmissionMatrix
    {
        public static MathNet.Numerics.LinearAlgebra.Vector<Complex> GenerateTransmissionVector(int size)
        {
            // The transmission vector is normalized according to Vellekoop thesis (p. 93).
            // Normalization factor is 1 / sqrt(size). Additional factor 1 / sqrt(2) appears
            // due to construction of the transmission vector of two Normal random numbers.
            MathNet.Numerics.LinearAlgebra.Vector<Complex> tm =
                MathNet.Numerics.LinearAlgebra.Vector<Complex>.Build.Dense(size);
            double denom = Math.Sqrt(2.0 * size);
            for (int i = 0; i < size; i++)
            {
                tm[i] = new Complex(Normal.Sample(0.0, 1.0) / denom,
                    Normal.Sample(0.0, 1.0) / denom);
            }
            return tm;
        }

        public static MathNet.Numerics.LinearAlgebra.Vector<Complex> GenerateTransmissionVector(int size, bool align_to_x)
        {
            MathNet.Numerics.LinearAlgebra.Vector<Complex> t_vector = GenerateTransmissionVector(size);
            Complex align_multiplier = t_vector.Sum().Conjugate();
            align_multiplier /= align_multiplier.Magnitude;
            return t_vector * align_multiplier;
        }

        public static MathNet.Numerics.LinearAlgebra.Vector<Complex>[] GenerateTransmissionVectorsArray(int size, int length, bool align)
        {
            MathNet.Numerics.LinearAlgebra.Vector<Complex>[] t_vectors =
                new MathNet.Numerics.LinearAlgebra.Vector<Complex>[length];

            for (int i = 0; i < length; i++)
                t_vectors[i] = GenerateTransmissionVector(size, align);

            return t_vectors;
        }
    }
}
