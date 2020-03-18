using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Linq;

using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;
using MathNet.Numerics.Distributions;

namespace HadamardWienerFilter
{
    public static class HadamardAlgorithm
    {
        public static HadamardResult Simulate(MathNet.Numerics.LinearAlgebra.Vector<Complex> t_vector,
            Complex e_inc, Func<double, double> detector, bool avoid_zero_i_1 = false)
        {
            Matrix<Complex> h_matrix = HadamardMartix.GenerateHadamardMatrix(t_vector.Count);
            HadamardResult h_res = new HadamardResult();

            int slm_size = t_vector.Count;
            h_res.IntensityMinus = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(slm_size);
            h_res.IntensityPlus = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(slm_size);
            h_res.ReSigma = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(slm_size);
            h_res.TransmissionVectorEstimated = MathNet.Numerics.LinearAlgebra.Vector<Complex>.Build.Dense(slm_size);
            h_res.SLMPatternOptimized = MathNet.Numerics.LinearAlgebra.Vector<Complex>.Build.Dense(slm_size);
            h_res.Zeta = MathNet.Numerics.LinearAlgebra.Vector<Complex>.Build.Dense(2 * slm_size - 1);

            MathNet.Numerics.LinearAlgebra.Vector<Complex> h_1 = h_matrix.Row(0);
            double i_1 = detector(((t_vector * e_inc) * h_1).MagnitudeSquared());

            if (avoid_zero_i_1 && i_1 == 0.0)
            {
                h_res.ZeroInitialIntensity = true;
                i_1 = 1.0;
                Console.WriteLine("I_1 was equal to zero.");
            }

            double i_plus;
            double i_minus;
            double re_sigma;

            for (int pix = 0; pix < t_vector.Count; pix++)
            {
                MathNet.Numerics.LinearAlgebra.Vector<Complex> h_i = h_matrix.Row(pix);
                MathNet.Numerics.LinearAlgebra.Vector<Complex> v_plus = (h_1 + h_i) / 2.0;
                MathNet.Numerics.LinearAlgebra.Vector<Complex> v_minus = (h_1 - h_i) / 2.0;

                i_plus = detector(((t_vector * e_inc) * v_plus).MagnitudeSquared());
                i_minus = detector(((t_vector * e_inc) * v_minus).MagnitudeSquared());
                h_res.IntensityPlus[pix] = i_plus;
                h_res.IntensityMinus[pix] = i_minus;

                h_res.Zeta[pix] = i_plus / Math.Sqrt(i_1) / t_vector.Count;
                if (pix != 0)
                    h_res.Zeta[slm_size + pix - 1] = i_minus / Math.Sqrt(i_1) / t_vector.Count;


                re_sigma = (i_plus - i_minus) / Math.Sqrt(i_1) / t_vector.Count;
                h_res.ReSigma[pix] = re_sigma;

                h_res.TransmissionVectorEstimated += re_sigma * h_i;
            }

            for (int i = 0; i < slm_size; i++)
            {
                if (h_res.TransmissionVectorEstimated[i].Real >= 0)
                    h_res.SLMPatternOptimized[i] = new Complex(1, 0);
                else
                    h_res.SLMPatternOptimized[i] = new Complex(0, 0);
            }

            for (int i = 0; i < slm_size; i++)
            {
                if (Math.Sign(h_res.TransmissionVectorEstimated[i].Real) == Math.Sign(t_vector[i].Real))
                    h_res.CorrectSignsNumber++;
            }

            //h_res.Enhancement = detector(((t_vector * e_inc) * h_res.SLMPatternOptimized).MagnitudeSquared()) /
            //    detector(((t_vector * e_inc) * h_1).MagnitudeSquared());
            h_res.OptimizedIntensity = detector(((t_vector * e_inc) * h_res.SLMPatternOptimized).MagnitudeSquared());
            h_res.Enhancement = h_res.OptimizedIntensity / i_1;

            return h_res;
        }
        public static HadamardResult[] BatchSimulate(Func<double, double> detector,
            MathNet.Numerics.LinearAlgebra.Vector<Complex>[] t_vectors, Complex e_inc, bool avoid_zero_i_1 = false)
        {
            // Prepare variables for filter estimation
            int slm_size = t_vectors[0].Count;
            int t_vecs_count = t_vectors.Length;
            Matrix<Complex> h_mat = HadamardMartix.GenerateHadamardMatrix(slm_size);
            HadamardResult[] h_results = new HadamardResult[t_vecs_count];

            // Hadamard algorithm simulation for Wiener filter generation.
            for (int i = 0; i < t_vectors.Length; i++)
                h_results[i] = Simulate(t_vectors[i], e_inc, detector, avoid_zero_i_1);

            return h_results;
        }

        public static void BatchStatistics(HadamardResult[] h_results)
        {
            // Average number of correct estimations
            double correct_estimations =
                h_results.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));
            double average_signal =
                h_results.Average(new Func<HadamardResult, double>(e => (e.IntensityPlus.Average() + e.IntensityMinus.Average()) / 2.0));
            double maximal_signal =
                h_results.Average(new Func<HadamardResult, double>(e => Math.Max(e.IntensityPlus.Max(), e.IntensityMinus.Max())));
            double minimal_signal =
                h_results.Average(new Func<HadamardResult, double>(e => Math.Max(e.IntensityPlus.Min(), e.IntensityMinus.Min())));
            double average_opt_signal = h_results.Average(new Func<HadamardResult, double>(e => e.OptimizedIntensity));
        }
    }
}
