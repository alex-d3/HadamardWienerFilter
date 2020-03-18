using System;
using System.Numerics;
using System.Linq;
using System.Collections.Generic;

using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using mn_linalg = MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;
using MathNet.Numerics.Distributions;

namespace HadamardWienerFilter
{
    public class WienerFilter
    {
        private double gamma_zeta_cond;
        private MathNet.Numerics.LinearAlgebra.Vector<double> gamma_zeta_eig;

        private Matrix<double> gamma_zeta_tau;
        private Matrix<double> gamma_zeta;
        private Matrix<double> gamma_zeta_inverse;
        private Matrix<double> G_inv;

        public Matrix<double> Gamma
        {
            get { return G_inv; }
        }

        public Matrix<double> GammaZeta
        {
            get { return gamma_zeta; }
        }

        public MathNet.Numerics.LinearAlgebra.Vector<double> GammaZetaEig
        {
            get
            {
                if (gamma_zeta_eig == null)
                    gamma_zeta_eig = gamma_zeta.Evd().EigenValues.Real();
                return gamma_zeta_eig;
            }
        }

        public double GammaZetaCond
        {
            get
            {
                if (gamma_zeta_cond == 0.0)
                    gamma_zeta_cond = gamma_zeta.ConditionNumber();
                return gamma_zeta_cond;
            }
        }

        public WienerFilter(HadamardResult[] h_results,
            MathNet.Numerics.LinearAlgebra.Vector<Complex>[] t_vectors)
        {
            //List<HadamardResult> h_results_filtered = new List<HadamardResult>(h_results.Length);
            //List<mn_linalg::Vector<Complex>> t_vectors_filtered =
            //    new List<mn_linalg::Vector<Complex>>(t_vectors.Length);

            //for (int i = 0; i < h_results.Length; i++)
            //{
            //    if (!h_results[i].ZeroInitialIntensity)
            //    {
            //        h_results_filtered.Add(h_results[i]);
            //        t_vectors_filtered.Add(t_vectors[i]);
            //    }
            //}
            //Console.WriteLine("Only {0} valid simulations will be used for the filter construction.",
            //    h_results_filtered.Count);

            //Build(h_results_filtered.ToArray(), t_vectors_filtered.ToArray());
            Build(h_results, t_vectors);
        }
        private HadamardResult[] FilterResults(HadamardResult[] h_results)
        {
            List<HadamardResult> h_results_filtered = new List<HadamardResult>(h_results.Length);

            for (int i = 0; i < h_results.Length; i++)
            {
                if (!h_results[i].ZeroInitialIntensity)
                    h_results_filtered.Add(h_results[i]);
            }
            Console.WriteLine("Only {0} valid simulations will be used for the filter construction.",
                h_results_filtered.Count);
            return h_results_filtered.ToArray();
        }
        private void Build(HadamardResult[] h_results,
            MathNet.Numerics.LinearAlgebra.Vector<Complex>[] t_vectors)
        {
            // Prepare variables for filter estimation
            int slm_size = t_vectors[0].Count;
            int t_vecs_count = t_vectors.Length;

            // Estimate Gamma_zeta matrix
            gamma_zeta = Matrix<double>.Build.Dense(2 * slm_size - 1, 2 * slm_size - 1);
            for (int n = 0; n < t_vecs_count; n++)
                gamma_zeta += h_results[n].Zeta.OuterProduct(h_results[n].Zeta).Real();
            gamma_zeta /= t_vecs_count;
            // Properties of Gamma_zeta matrix
            //gamma_zeta_eig = gamma_zeta.Evd().EigenValues.Real();
            //gamma_zeta_cond = gamma_zeta.ConditionNumber();

            // Estimate Gamma_zeta*tau matrix
            gamma_zeta_tau = Matrix<double>.Build.Dense(2 * slm_size - 1, slm_size);
            for (int n = 0; n < t_vecs_count; n++)
                gamma_zeta_tau += h_results[n].Zeta.Real().OuterProduct(t_vectors[n].Real());
            gamma_zeta_tau /= t_vecs_count;

            // Estimation of the G matrix
            gamma_zeta_inverse = gamma_zeta.Inverse();
            //Matrix<double> unity_matrix = gamma_zeta_inverse * gamma_zeta;
            //Matrix<double> unity_matrix_residual = unity_matrix - Matrix<double>.Build.DenseIdentity(gamma_zeta_inverse.RowCount);
            //double unity_matrix_residual_abs_sum = unity_matrix_residual.ColumnAbsoluteSums().Sum();
            G_inv = gamma_zeta_inverse * gamma_zeta_tau;
        }

        public MathNet.Numerics.LinearAlgebra.Vector<double> Apply(MathNet.Numerics.LinearAlgebra.Vector<double> zeta)
        {
            return zeta * G_inv;
        }

        public MathNet.Numerics.LinearAlgebra.Vector<double>[] BatchApply(MathNet.Numerics.LinearAlgebra.Vector<double>[] zeta)
        {
            MathNet.Numerics.LinearAlgebra.Vector<double>[] t_vectors_est =
                new MathNet.Numerics.LinearAlgebra.Vector<double>[zeta.Length];
            for (int i = 0; i < t_vectors_est.Length; i++)
                t_vectors_est[i] = Apply(zeta[i]);
            return t_vectors_est;
        }
    }
}
