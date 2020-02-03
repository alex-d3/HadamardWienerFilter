using System;
using System.Collections.Generic;
using System.Text;

using System.Numerics;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;

namespace HadamardWienerFilter
{
    internal class HadamardMartix
    {
        private static Matrix<Complex> h_mat;
        private static Matrix<Complex> h1;

        public static Matrix<Complex> GenerateHadamardMatrix(int channels)
        {
            int power = Log2(channels);

            //if (power < 1 || power > (sizeof(int) * 8))
            //    throw new ArgumentException("Improper matrix order.");

            if (h_mat == null || h_mat.RowCount != power)
            {
                h1 = Matrix<Complex>.Build.Dense(2, 2,
                new Complex[] { 1.0, 1.0, 1.0, -1.0 });
                Matrix<Complex> h = h1.Clone();

                for (int i = 1; i < power; i++)
                    h = h1.KroneckerProduct(h);
                return h;
            }
            else
                return h_mat;
        }

        private static int Log2(int num)
        {
            if ((num & (~num)) != 0)
            {
                throw new ArgumentException(
                    string.Format("Argument {0} = {1} has to be a power of 2.", nameof(num), num));
            }
            int power = 0;
            while ((num >>= 1) > 0)
                ++power;
            return power;
        }
    }
}
