using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Globalization;
using System.IO;

using CsvHelper;
using MathNet.Numerics.LinearAlgebra;

namespace HadamardWienerFilter
{
    public static class VectorExtensions
    {
        public static void WriteToCSV(this MathNet.Numerics.LinearAlgebra.Vector<Complex> vec, string path)
        {
            CsvHelper.Configuration.Configuration cfg =
                new CsvHelper.Configuration.Configuration(CultureInfo.InvariantCulture);

            using (StreamWriter sw = new StreamWriter(path))
            using (CsvWriter cw = new CsvWriter(sw, cfg))
            {
                for (int e = 0; e < vec.Count; e++)
                {
                    cw.WriteField(e);
                    cw.WriteField(vec[e].Real);
                    cw.WriteField(vec[e].Imaginary);
                    cw.NextRecord();
                }
            }
        }
        public static void WriteToCSV(this MathNet.Numerics.LinearAlgebra.Vector<double> vec, string path)
        {
            CsvHelper.Configuration.Configuration cfg =
                new CsvHelper.Configuration.Configuration(CultureInfo.InvariantCulture);

            using (StreamWriter sw = new StreamWriter(path))
            using (CsvWriter cw = new CsvWriter(sw, cfg))
            {
                for (int e = 0; e < vec.Count; e++)
                {
                    cw.WriteField(e);
                    cw.WriteField(vec[e]);
                    cw.NextRecord();
                }
            }
        }
    }
}
