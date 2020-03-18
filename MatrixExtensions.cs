using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Globalization;
using System.Numerics;

using CsvHelper;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace HadamardWienerFilter
{
    public static class MatrixExtensions
    {
        public static void WriteToCSV(this Matrix<Complex> mat, string path)
        {
            CsvHelper.Configuration.Configuration cfg =
                new CsvHelper.Configuration.Configuration(CultureInfo.InvariantCulture);

            using (StreamWriter sw = new StreamWriter(path))
            using (CsvWriter cw = new CsvWriter(sw, cfg))
            {
                for (int row = 0; row < mat.RowCount; row++)
                {
                    for (int col = 0; col < mat.ColumnCount; col++)
                    {
                        cw.WriteField(row);
                        cw.WriteField(col);
                        cw.WriteField(mat[row, col].Real);
                        cw.WriteField(mat[row, col].Imaginary);
                        cw.NextRecord();
                    }
                }

            }
        }
        public static void WriteToCSV(this Matrix<double> mat, string path)
        {
            CsvHelper.Configuration.Configuration cfg =
                new CsvHelper.Configuration.Configuration(CultureInfo.InvariantCulture);

            using (StreamWriter sw = new StreamWriter(path))
            using (CsvWriter cw = new CsvWriter(sw, cfg))
            {
                for (int row = 0; row < mat.RowCount; row++)
                {
                    for (int col = 0; col < mat.ColumnCount; col++)
                    {
                        cw.WriteField(row);
                        cw.WriteField(col);
                        cw.WriteField(mat[row, col]);
                        cw.NextRecord();
                    }
                }

            }
        }
        public static void WriteToBinary(this Matrix<double> mat, string path)
        {
            using (BinaryWriter bw =
                    new BinaryWriter(new FileStream(path, FileMode.Create)))
            {
                double[] mat_array = mat.AsColumnMajorArray();
                for (int row = 0; row < mat.RowCount; row++)
                    for (int col = 0; col < mat.ColumnCount; col++)
                        bw.Write(mat_array[col + row * mat.ColumnCount]);
            }
        }
    }
}
