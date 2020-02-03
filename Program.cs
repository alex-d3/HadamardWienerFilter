using System;
using System.Numerics;
using System.IO;
using System.Globalization;
using System.Linq;

using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;
using MathNet.Numerics.Distributions;
using CsvHelper;

using ImageSensorSimulator;

namespace HadamardWienerFilter
{
    class Program
    {
        static ImageSensor OnePixelSensor()
        {
            ImageSensorProperties properties = new ImageSensorProperties();

            properties.ImageSensorRows = 1;
            properties.ImageSensorColumns = 1;

            properties.PDET_PitchX = 5.0e-6;
            properties.PDET_PitchY = 5.0e-6;

            properties.SIMFL_PhotonShotNoise = true;
            properties.SIMFL_PhotoelectronsShotNoise = true;

            properties.SensorKind = SensorKind.CCD;

            properties.Temperature = 300.0;

            properties.PDET_FillFactor = 0.95;

            properties.IntegrationTime = 0.035;
            properties.PDET_ExtQuantumEff = 0.8;
            properties.PDET_QuantumYield = 1.0;

            properties.SIMFL_PhotoResponseNonUniformity = true;
            properties.PDET_Light_PRNU_Model = NoiseDistribution.JanesickGaussian;
            properties.PDET_Light_PRNU_Sigma = 0.01;
            // properties.PDET_Light_PRNU_Sigma = 0.05; // Experimental

            properties.M_BandGap_0 = 1.166;
            properties.M_VarshniAlpha = 5.5e-04;
            properties.M_VarshniBeta = 636.0;

            properties.SIMFL_DarkCurrent = true;
            //properties.PDET_DarkCurrentFigOfMerit = 1.0 * 1.0e-9 * 1.0e4;
            properties.PDET_DarkCurrentFigOfMerit = 1.0 * 1.0e-9 * 1.0e4 * 2.0e1;
            properties.PDET_DarkCurrentCoef_ca = 4.31e5;
            properties.PDET_DarkCurrentCoef_ed = 2.0;

            properties.SIMFL_DarkCurrentShotNoise = true;
            // There was a Gaussian noise model in the original code but it was not used.

            properties.SIMFL_DarkCurrentFPN = true;
            properties.PDET_Dark_PRNU_Model = NoiseDistribution.LogNormal;
            // properties.PDET_Dark_FPN_Sigma = 0.4;
            properties.PDET_Dark_FPN_Sigma = 0.8;

            properties.SN_Gain = 5.0e-6;
            properties.SN_V_ref_Reset = 3.1;
            properties.SN_Vs_min = 0.5;
            properties.SIMFL_SenseNodeNonlinearGainResponse = false;
            properties.SN_GainResponse_k1 = 1.090900000e-14;
            properties.SN_FullEellElectrons = 2.0e4;

            properties.SIMFL_SenseNodeResetNoise = true;
            properties.SN_ResetNoiseFactor = 0.8;
            // There was a Gaussian reset noise model but it was not used anywhere.

            properties.SF_Gain = 1.0;

            properties.SIMFL_SourceFollowerNonLinearity = true;
            properties.SF_Nonlinearity_Ratio = 1.05;
            properties.SF_Flicker_Corner_Freq = 1.0e6;
            properties.SF_White_Noise_Density = 15.0e-9;
            properties.SF_Delta_Induced_Modulation = 1.0e-8;
            properties.SF_Data_Clock_Speed = 20.0e6;
            properties.SF_Freq_Sampling_Delta = 10000.0;
            properties.SIMFL_SourceFollowerNoise = true;

            properties.SIMFL_SourceFollowerFixedPatternOffset = true;
            properties.SF_Fixed_Pattern_Offset_Model = NoiseDistribution.JanesickGaussian;
            properties.SF_Fixed_Pattern_Offset_Sigma = 0.0005;

            properties.SF_CDS_Sample_to_Sampling_Time = 1.0e-6;
            properties.SF_CDS_Gain = 1.0;

            properties.ADC_Resolution = 12;
            properties.ADC_Offset = 0.0;
            properties.SIMFL_ADC_Nonlinearity = false;
            properties.ADC_NonlinearityRatio = 1.1;

            properties.PGA_Gain = 2.0;

            ImageSensor sensor = new ImageSensor(properties);

            return sensor;
        }

        static ImageSensor OnePixelSensorNoiseless()
        {
            ImageSensorProperties properties = new ImageSensorProperties();

            properties.ImageSensorRows = 1;
            properties.ImageSensorColumns = 1;

            properties.PDET_PitchX = 5.0e-6;
            properties.PDET_PitchY = 5.0e-6;

            properties.SIMFL_PhotonShotNoise = false;
            properties.SIMFL_PhotoelectronsShotNoise = false;

            properties.SensorKind = SensorKind.CCD;

            properties.Temperature = 300.0;

            properties.PDET_FillFactor = 0.95;

            properties.IntegrationTime = 0.035;
            properties.PDET_ExtQuantumEff = 0.8;
            properties.PDET_QuantumYield = 1.0;

            properties.SIMFL_PhotoResponseNonUniformity = false;
            properties.PDET_Light_PRNU_Model = NoiseDistribution.JanesickGaussian;
            properties.PDET_Light_PRNU_Sigma = 0.01;
            // properties.PDET_Light_PRNU_Sigma = 0.05; // Experimental

            properties.M_BandGap_0 = 1.166;
            properties.M_VarshniAlpha = 5.5e-04;
            properties.M_VarshniBeta = 636.0;

            properties.SIMFL_DarkCurrent = false;
            properties.PDET_DarkCurrentFigOfMerit = 1.0 * 1.0e-9 * 1.0e4;
            properties.PDET_DarkCurrentCoef_ca = 4.31e5;
            properties.PDET_DarkCurrentCoef_ed = 2.0;

            properties.SIMFL_DarkCurrentShotNoise = false;
            // There was a Gaussian noise model in the original code but it was not used.

            properties.SIMFL_DarkCurrentFPN = false;
            properties.PDET_Dark_PRNU_Model = NoiseDistribution.LogNormal;
            properties.PDET_Dark_FPN_Sigma = 0.4;

            properties.SN_Gain = 5.0e-6;
            properties.SN_V_ref_Reset = 3.1;
            properties.SN_Vs_min = 0.5;
            properties.SIMFL_SenseNodeNonlinearGainResponse = false;
            properties.SN_GainResponse_k1 = 1.090900000e-14;
            properties.SN_FullEellElectrons = 2.0e4;

            properties.SIMFL_SenseNodeResetNoise = false;
            properties.SN_ResetNoiseFactor = 0.8;
            // There was a Gaussian reset noise model but it was not used anywhere.

            properties.SF_Gain = 1.0;

            properties.SIMFL_SourceFollowerNonLinearity = false;
            properties.SF_Nonlinearity_Ratio = 1.05;
            properties.SF_Flicker_Corner_Freq = 1.0e6;
            properties.SF_White_Noise_Density = 15.0e-9;
            properties.SF_Delta_Induced_Modulation = 1.0e-8;
            properties.SF_Data_Clock_Speed = 20.0e6;
            properties.SF_Freq_Sampling_Delta = 10000.0;
            properties.SIMFL_SourceFollowerNoise = false;

            properties.SIMFL_SourceFollowerFixedPatternOffset = false;
            properties.SF_Fixed_Pattern_Offset_Model = NoiseDistribution.JanesickGaussian;
            properties.SF_Fixed_Pattern_Offset_Sigma = 0.0005;

            properties.SF_CDS_Sample_to_Sampling_Time = 1.0e-6;
            properties.SF_CDS_Gain = 1.0;

            properties.ADC_Resolution = 8;
            properties.ADC_Offset = 0.0;
            properties.SIMFL_ADC_Nonlinearity = false;
            properties.ADC_NonlinearityRatio = 1.1;

            properties.PGA_Gain = 1.0;

            ImageSensor sensor = new ImageSensor(properties);

            return sensor;
        }

        static ImageSensor CustomSensor(
            int rows, int cols, double temperature = 323.15, double pga_gain = 1.0, double exposure_time = 0.035)
        {
            ImageSensorProperties properties = new ImageSensorProperties();

            properties.ImageSensorRows = rows;
            properties.ImageSensorColumns = cols;

            properties.PDET_PitchX = 5.0e-6;
            properties.PDET_PitchY = 5.0e-6;

            properties.SIMFL_PhotonShotNoise = true;
            properties.SIMFL_PhotoelectronsShotNoise = true;

            properties.SensorKind = SensorKind.CCD;

            properties.Temperature = temperature;

            properties.PDET_FillFactor = 0.95;

            properties.IntegrationTime = exposure_time;
            properties.PDET_ExtQuantumEff = 0.8;
            properties.PDET_QuantumYield = 1.0;

            properties.SIMFL_PhotoResponseNonUniformity = true;
            properties.PDET_Light_PRNU_Model = NoiseDistribution.JanesickGaussian;
            properties.PDET_Light_PRNU_Sigma = 0.01;
            // properties.PDET_Light_PRNU_Sigma = 0.05; // Experimental

            properties.M_BandGap_0 = 1.166;
            properties.M_VarshniAlpha = 5.5e-04;
            properties.M_VarshniBeta = 636.0;

            properties.SIMFL_DarkCurrent = true;
            properties.PDET_DarkCurrentFigOfMerit = 1.0 * 1.0e-9 * 1.0e4;
            //properties.PDET_DarkCurrentFigOfMerit = 1.0 * 1.0e-9 * 1.0e4 * 2.0e1;
            properties.PDET_DarkCurrentCoef_ca = 4.31e5;
            properties.PDET_DarkCurrentCoef_ed = 2.0;

            properties.SIMFL_DarkCurrentShotNoise = true;
            // There was a Gaussian noise model in the original code but it was not used.

            properties.SIMFL_DarkCurrentFPN = true;
            properties.PDET_Dark_PRNU_Model = NoiseDistribution.LogNormal;
            properties.PDET_Dark_FPN_Sigma = 0.4;
            //properties.PDET_Dark_FPN_Sigma = 0.8;

            properties.SN_Gain = 5.0e-6;
            properties.SN_V_ref_Reset = 3.1;
            properties.SN_Vs_min = 0.5;
            properties.SIMFL_SenseNodeNonlinearGainResponse = true;
            properties.SN_GainResponse_k1 = 1.090900000e-14;
            properties.SN_FullEellElectrons = 2.0e4;

            properties.SIMFL_SenseNodeResetNoise = true;
            properties.SN_ResetNoiseFactor = 0.8;
            // There was a Gaussian reset noise model but it was not used anywhere.

            properties.SF_Gain = 1.0;

            properties.SIMFL_SourceFollowerNonLinearity = true;
            properties.SF_Nonlinearity_Ratio = 1.05;
            properties.SF_Flicker_Corner_Freq = 1.0e6;
            properties.SF_White_Noise_Density = 15.0e-9;
            properties.SF_Delta_Induced_Modulation = 1.0e-8;
            properties.SF_Data_Clock_Speed = 20.0e6;
            properties.SF_Freq_Sampling_Delta = 10000.0;
            properties.SIMFL_SourceFollowerNoise = true;

            properties.SIMFL_SourceFollowerFixedPatternOffset = true;
            properties.SF_Fixed_Pattern_Offset_Model = NoiseDistribution.JanesickGaussian;
            properties.SF_Fixed_Pattern_Offset_Sigma = 0.0005;

            properties.SF_CDS_Sample_to_Sampling_Time = 1.0e-6;
            properties.SF_CDS_Gain = 1.0;

            properties.ADC_Resolution = 8;
            properties.ADC_Offset = 0.0;
            properties.SIMFL_ADC_Nonlinearity = false; // was always false
            properties.ADC_NonlinearityRatio = 1.1;

            properties.PGA_Gain = pga_gain;

            ImageSensor sensor = new ImageSensor(properties);

            return sensor;
        }

        static double SimulateCamera(ImageSensor sensor, double input, string save_to = null)
        {
            Matrix<double> sensor_input = Matrix<double>.Build.Dense(sensor.Rows, sensor.Columns);
            double[] sensor_input_arr = sensor_input.AsColumnMajorArray();
            for (int k = 0; k < sensor_input_arr.Length; k++)
                sensor_input_arr[k] = input;
            Matrix<double> sensor_result = sensor.Simulate(sensor_input);
            //double[] sensor_output_arr = sensor_result.AsColumnMajorArray();
            //double output = 0.0;
            //for (int k = 0; k < sensor_output_arr.Length; k++)
            //    output += sensor_output_arr[k];
            //output /= sensor_output_arr.Length;

            //if (save_to != null)
            //{
            //    using (BinaryWriter bw =
            //        new BinaryWriter(new FileStream(save_to, FileMode.Create)))
            //    {
            //        for (int i = 0; i < sensor_output_arr.Length; i++)
            //        {
            //            bw.Write(sensor_output_arr[i]);
            //        }
            //    }
            //}
            //return output;
            return sensor_result[0, 0];
        }

        public static HadamardResult Simulate(
            MathNet.Numerics.LinearAlgebra.Vector<Complex> t_vector,
            Matrix<Complex> h_matrix,
            Func<double, double> detector, bool avoid_zero_i_1 = false)
        {
            HadamardResult h_res = new HadamardResult();

            int slm_size = t_vector.Count;
            h_res.IntensityMinus = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(slm_size);
            h_res.IntensityPlus = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(slm_size);
            h_res.ReSigma = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(slm_size);
            h_res.TransmissionVectorEstimated = MathNet.Numerics.LinearAlgebra.Vector<Complex>.Build.Dense(slm_size);
            h_res.SLMPatternOptimized = MathNet.Numerics.LinearAlgebra.Vector<Complex>.Build.Dense(slm_size);
            h_res.Zeta = MathNet.Numerics.LinearAlgebra.Vector<Complex>.Build.Dense(2 * slm_size - 1);

            MathNet.Numerics.LinearAlgebra.Vector<Complex> h_1 = h_matrix.Row(0);
            double i_1 = detector((t_vector * h_1).MagnitudeSquared());

            if (avoid_zero_i_1 && i_1 == 0.0)
            {
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

                i_plus = detector((t_vector * v_plus).MagnitudeSquared());
                i_minus = detector((t_vector * v_minus).MagnitudeSquared());
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

            return h_res;
        }

        static void SaveMatrix(Matrix<Complex> mat, string path)
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

        static void SaveMatrix(Matrix<double> mat, string path)
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

        static void SaveVector(MathNet.Numerics.LinearAlgebra.Vector<double> vec, string path)
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

        static void SaveVector(MathNet.Numerics.LinearAlgebra.Vector<Complex> vec, string path)
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

        static void SaveMatrixBinary(Matrix<double> mat, string path)
        {
            using (BinaryWriter bw =
                    new BinaryWriter(new FileStream(path, FileMode.Create)))
            {
                double[] mat_array = mat.AsColumnMajorArray();

                //for (int i = 0; i < mat_array.Length; i++)
                //{
                //    bw.Write(mat_array[i]);
                //}
                for (int row = 0; row < mat.RowCount; row++)
                {
                    for (int col = 0; col < mat.ColumnCount; col++)
                    {
                        bw.Write(mat_array[col + row * mat.ColumnCount]);
                    }
                }
            }

        }

        static void TestSensor(string path)
        {
            ImageSensor sensor = CustomSensor(512, 512);
            Matrix<double> test_target = Matrix<double>.Build.Dense(512, 512);

            int fringe_width = 1;
            int col_start = 1, col_end = col_start + fringe_width;
            bool end = false;
            while (!end)
            {
                for (int row = 0; row < test_target.RowCount / 2; row++)
                {
                    for (int col = col_start; col < test_target.ColumnCount && col < col_end; col++)
                    {
                        test_target[row, col] = 1.0;
                    }
                }
                for (int row = test_target.RowCount / 2; row < test_target.RowCount; row++)
                {
                    for (int col = col_start; col < test_target.ColumnCount && col < col_end; col++)
                    {
                        test_target[row, col] = ((double)col) / test_target.ColumnCount;
                    }
                }
                fringe_width++;
                col_start += 2 * fringe_width;
                col_end = col_start + fringe_width;

                if (col_end >= test_target.ColumnCount)
                    end = true;
            }

            //for (int col = 0, fringe_width = 1; col < test_target.ColumnCount; col += 2 * fringe_width, fringe_width++)
            //{
            //    for (int row = 0; row < test_target.RowCount && (col + 2 * fringe_width) < test_target.ColumnCount; row++)
            //    {
            //        test_target[row, col + fringe_width] = 1.0;
            //    }
            //}

            test_target *= 2.0e16 * 0.10;

            test_target = test_target.Transpose();

            Matrix<double> image = sensor.Simulate(test_target);
            SaveMatrixBinary(image, path);
        }

        static void Main(string[] args)
        {
            //for (int i = 0; i < 1; i++)
            //{
            //    TestSensor(string.Format("d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/demo_sensor/test_frame_at_50C_{0}.bin", i));
            //}
            //return;
            //MathNet.Numerics.LinearAlgebra.Vector<double> test = MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(255);
            //Matrix<double> test_mat = Matrix<double>.Build.Dense(255, 255, 0.0);
            //for (int k = 0; k < 100; k++)
            //{
            //    for (int i = 0; i < test.Count; i++)
            //    {
            //        double sine = Math.Sin(i * 2.0 * Math.PI / test.Count) * 1.0e10 + 1.2e10;
            //        test[i] = Normal.Sample(sine, sine);

            //        //test[i] = Normal.Sample(10.0, 1.0) + 1.0;

            //        //double sine = Math.Sin(i * 2.0 * Math.PI / test.Count) * 100.0 + 105.0;
            //        //test[i] = Poisson.Sample(sine) + Normal.Sample(0.0, 2.0);
            //    }
            //    test_mat += test.ToColumnMatrix() * test.ToRowMatrix();
            //}
            //test_mat /= 100.0;
            //Console.WriteLine(test_mat);
            ////for (int i = 0; i < 127; i++)
            ////    test_array[i] = test_array[i + 127];
            //Console.WriteLine(test_mat.Determinant());
            //return;

            // Prepare variables for filter estimation
            Matrix<Complex> h_mat;
            HadamardResult[] h_results_ideal;
            HadamardResult[] h_results_cam;
            HadamardResult[] h_results_cam_noiseless;

            HadamardResult[] h_results_test_ideal;
            HadamardResult[] h_results_test_cam;
            HadamardResult[] h_results_test_cam_noiseless;

            //ImageSensor sensor = OnePixelSensor();
            ImageSensor sensor = CustomSensor(1, 1, 325.15, 2.0, 0.035);
            ImageSensor sensor_noiseless = OnePixelSensorNoiseless();

            int slm_size = 64;

            int t_vecs_count = 1024;

            h_mat = HadamardMartix.GenerateHadamardMatrix(slm_size);

            // Create a set of transmission matrices for Wiener filter calculation
            MathNet.Numerics.LinearAlgebra.Vector<Complex>[] t_vectors =
                TransmissionMatrix.GenerateTransmissionVectorsArray(slm_size, t_vecs_count, true);

            if (slm_size == 4)
            {
                for (int i = 0; i < t_vectors.Length; i++)
                    SaveVector(t_vectors[i],
                        Path.Combine(Path.GetTempPath(), string.Format("{0}_{1}.csv", nameof(t_vectors), i)));
            }

            // Create a set of transmission matrices for Wiener filter testing
            MathNet.Numerics.LinearAlgebra.Vector<Complex>[] t_vectors_test =
                TransmissionMatrix.GenerateTransmissionVectorsArray(slm_size, t_vecs_count, true);

            Console.WriteLine("Transmission vectors were prepared.");

            for (int i = 0; i < t_vectors.Length; i++)
            {
                double magnitude = t_vectors[i].Sum().Magnitude;
                t_vectors[i] /= magnitude;
                double magnitude_after = t_vectors[i].Sum().Magnitude;
            }

            Console.WriteLine("Transmission vectors were normalized to 1.");

            // Create arrays for results
            h_results_ideal = new HadamardResult[t_vecs_count];
            h_results_cam = new HadamardResult[t_vecs_count];
            h_results_cam_noiseless = new HadamardResult[t_vecs_count];

            h_results_test_ideal = new HadamardResult[t_vecs_count];
            h_results_test_cam = new HadamardResult[t_vecs_count];
            h_results_test_cam_noiseless = new HadamardResult[t_vecs_count];

            // Create functions that calculate intensity directly and simulate an image sensor
            Func<double, double> simple_detector = new Func<double, double>((input_intensity) => {
                return input_intensity;
            });
            Func<double, double> advanced_detector = new Func<double, double>((input_intensity) => {
                return SimulateCamera(sensor, input_intensity * 2.0e16 * 0.05);
            });
            Func<double, double> advanced_detector_noiseless = new Func<double, double>((input_intensity) => {
                return SimulateCamera(sensor_noiseless, input_intensity * 2.0e16 * 0.05);
            });

            // Hadamard algorithm simulation for Wiener filter generation.
            Console.WriteLine("Calculating primary data.");
            for (int i = 0; i < t_vectors.Length; i++)
            {
                h_results_ideal[i] = Simulate(t_vectors[i], h_mat, simple_detector);
                h_results_cam[i] = Simulate(t_vectors[i], h_mat, advanced_detector, true);
                h_results_cam_noiseless[i] = Simulate(t_vectors[i], h_mat, advanced_detector_noiseless, true);
            }

            Console.WriteLine("Calculating test data.");
            for (int i = 0; i < t_vectors.Length; i++)
            {
                h_results_test_ideal[i] = Simulate(t_vectors_test[i], h_mat, simple_detector);
                h_results_test_cam[i] = Simulate(t_vectors_test[i], h_mat, advanced_detector, true);
                h_results_test_cam_noiseless[i] = Simulate(t_vectors_test[i], h_mat, advanced_detector_noiseless, true);
            }

            if (slm_size == 4)
            {
                for (int i = 0; i < h_results_cam.Length; i++)
                {
                    SaveVector(h_results_cam[i].Zeta,
                        Path.Combine(Path.GetTempPath(), string.Format("h_result_cam_zeta_{0}.csv", i)));
                    SaveVector(h_results_cam[i].TransmissionVectorEstimated,
                        Path.Combine(Path.GetTempPath(), string.Format("h_result_cam_t_vec_est_{0}.csv", i)));
                }
            }

            // Average number of correct estimations
            double correct_estimations_ideal =
                h_results_ideal.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));
            double correct_estimations_cam =
                h_results_cam.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));
            double correct_estimations_cam_noiseless =
                h_results_cam_noiseless.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));

            double correct_estimations_test_ideal =
                h_results_test_ideal.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));
            double correct_estimations_test_cam =
                h_results_test_cam.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));
            double correct_estimations_test_cam_noiseless =
                h_results_test_cam_noiseless.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));

            MathNet.Numerics.LinearAlgebra.Vector<double> noise_cam =
                MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(h_results_cam_noiseless[0].IntensityPlus.Count);
            for (int i = 0; i < t_vecs_count; i++)
            {
                noise_cam += (h_results_cam[i].IntensityPlus - h_results_cam_noiseless[i].IntensityPlus).PointwiseAbs();
                noise_cam += (h_results_cam[i].IntensityMinus - h_results_cam_noiseless[i].IntensityMinus).PointwiseAbs();
            }
            noise_cam /= 2.0 * t_vecs_count;
            double noise_cam_avg = noise_cam.Sum() / noise_cam.Count;
            double noise_cam_min = noise_cam.Minimum();
            double noise_cam_max = noise_cam.Maximum();
            double signal_cam_noiseless_max =
                h_results_cam_noiseless.Max(new Func<HadamardResult, double>(
                    e => Math.Max(e.IntensityPlus.Maximum(), e.IntensityMinus.Maximum())));

            double noise_cam_average = 0.0;
            double noise_cam_stddev = 0.0;
            for (int i = 0; i < t_vecs_count; i++)
            {
                for (int j = 0; j < slm_size; j++)
                {
                    noise_cam_average += h_results_cam[i].IntensityPlus[j] - h_results_cam_noiseless[i].IntensityPlus[j];
                }
                // Console.WriteLine(h_results_cam[i].IntensityPlus[0] - h_results_cam_noiseless[i].IntensityPlus[0]);
            }
            noise_cam_average /= slm_size * t_vecs_count;
            for (int i = 0; i < t_vecs_count; i++)
                for (int j = 0; j < slm_size; j++)
                    noise_cam_stddev += Math.Pow(
                        h_results_cam[i].IntensityPlus[j] - h_results_cam_noiseless[i].IntensityPlus[j] - noise_cam_average, 2.0);
            noise_cam_stddev /= slm_size * t_vecs_count - 1;
            noise_cam_stddev = Math.Sqrt(noise_cam_stddev);

            //MathNet.Numerics.LinearAlgebra.Vector<double> zeta_mean =
            //    MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(h_results_cam[0].Zeta.Count);
            //MathNet.Numerics.LinearAlgebra.Vector<double> zeta_std =
            //    MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(h_results_cam[0].Zeta.Count);
            //MathNet.Numerics.LinearAlgebra.Vector<double> intensity_plus_mean =
            //    MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(h_results_cam[0].IntensityPlus.Length);
            //MathNet.Numerics.LinearAlgebra.Vector<double> intensity_plus_std =
            //    MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(h_results_cam[0].IntensityPlus.Length);
            //MathNet.Numerics.LinearAlgebra.Vector<double> intensity_minus_mean =
            //    MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(h_results_cam[0].IntensityMinus.Length);
            //MathNet.Numerics.LinearAlgebra.Vector<double> intensity_minus_std =
            //    MathNet.Numerics.LinearAlgebra.Vector<double>.Build.Dense(h_results_cam[0].IntensityMinus.Length);
            //for (int n = 0; n < t_vecs_count; n++)
            //{
            //    zeta_mean += h_results_cam[n].Zeta.Real();
            //}
            //zeta_mean /= t_vecs_count;
            //for (int n = 0; n < t_vecs_count; n++)
            //{
            //    zeta_std += (h_results_cam[n].Zeta.Real() - zeta_mean).PointwisePower(2.0);
            //}
            //zeta_std /= t_vecs_count - 1;
            //zeta_std = zeta_std.PointwiseSqrt();

            //SaveVector(zeta_mean, Path.Combine(Path.GetTempPath(), "zeta_mean.csv"));
            //SaveVector(zeta_std, Path.Combine(Path.GetTempPath(), "zeta_std.csv"));
            //ffdsfs
            // Estimate Gamma_zeta matrix
            Matrix<double> gamma_zeta_cam = Matrix<double>.Build.Dense(2 * slm_size - 1, 2 * slm_size - 1);
            double[] gamma_zeta_array = gamma_zeta_cam.AsColumnMajorArray();
            for (int n = 0; n < t_vecs_count; n++)
            {
                gamma_zeta_cam += h_results_cam[n].Zeta.OuterProduct(h_results_cam[n].Zeta).Real();
            }
            gamma_zeta_cam /= t_vecs_count;
            MathNet.Numerics.LinearAlgebra.Vector<double> gamma_zeta_cam_eig = gamma_zeta_cam.Evd().EigenValues.Real();

            // Gamma_zeta (only noise)
            Matrix<double> gamma_zeta_cam_only_noise = Matrix<double>.Build.Dense(2 * slm_size - 1, 2 * slm_size - 1);
            for (int n = 0; n < t_vecs_count; n++)
            {
                MathNet.Numerics.LinearAlgebra.Vector<Complex> noise = h_results_cam[n].Zeta - h_results_cam_noiseless[n].Zeta;
                gamma_zeta_cam_only_noise += noise.OuterProduct(noise).Real();
            }
            gamma_zeta_cam_only_noise /= t_vecs_count;
            MathNet.Numerics.LinearAlgebra.Vector<double> gamma_zeta_cam_only_noise_eig =
                gamma_zeta_cam_only_noise.Evd().EigenValues.Real();

            // Estimate Gamma_zeta*tau matrix
            Matrix<double> gamma_zeta_tau_cam = Matrix<double>.Build.Dense(2 * slm_size - 1, slm_size);
            double[] gamma_zeta_tau_array = gamma_zeta_tau_cam.AsColumnMajorArray();
            for (int n = 0; n < t_vecs_count; n++)
            {
                gamma_zeta_tau_cam += h_results_cam[n].Zeta.OuterProduct(h_results_ideal[n].TransmissionVectorEstimated).Real();
            }
            gamma_zeta_tau_cam /= t_vecs_count;

            //SaveVector(gamma_zeta_cam.Svd().S, Path.Combine(Path.GetTempPath(), "gamma_zeta_singularvalues.csv"));
            //SaveMatrix(gamma_zeta_cam, Path.Combine(Path.GetTempPath(), "gamma_zeta_cam.csv"));
            //SaveMatrix(gamma_zeta_tau_cam, Path.Combine(Path.GetTempPath(), "gamma_zeta_tau_cam.csv"));

            double gamma_zeta_cam_cond = gamma_zeta_cam.ConditionNumber();

            // Estimation of the G matrix
            Matrix<double> gamma_zeta_cam_inverse = gamma_zeta_cam.Inverse();
            Matrix<double> unity_matrix = gamma_zeta_cam_inverse * gamma_zeta_cam;
            Matrix<double> unity_matrix_residual = unity_matrix - Matrix<double>.Build.DenseIdentity(gamma_zeta_cam_inverse.RowCount);
            double unity_matrix_residual_abs_sum = unity_matrix_residual.ColumnAbsoluteSums().Sum();
            Matrix<double> gamma_zeta_pseudoinverse = gamma_zeta_cam.PseudoInverse();
            Matrix<double> G_inv = gamma_zeta_cam_inverse * gamma_zeta_tau_cam;
            Matrix<double> G_pinv = gamma_zeta_pseudoinverse * gamma_zeta_tau_cam;

            //SaveMatrix(gamma_zeta_inverse, Path.Combine(Path.GetTempPath(), "gamma_zeta_inverse.csv"));
            //SaveMatrix(gamma_zeta_pseudoinverse, Path.Combine(Path.GetTempPath(), "gamma_zeta_pseudoinverse.csv"));
            //SaveMatrix(G_inv, Path.Combine(Path.GetTempPath(), "G_inv.csv"));
            //SaveMatrix(G_pinv, Path.Combine(Path.GetTempPath(), "G_pinv.csv"));

            // Estimate the transmission matrix with a filter
            MathNet.Numerics.LinearAlgebra.Vector<double>[] t_vectors_filter_inv =
                new MathNet.Numerics.LinearAlgebra.Vector<double>[t_vecs_count];
            double correct_estimations_cam_opt = 0.0;
            for (int i = 0; i < t_vecs_count; i++)
            {
                MathNet.Numerics.LinearAlgebra.Vector<double> zeta_temp = h_results_cam[i].Zeta.Real();
                t_vectors_filter_inv[i] = zeta_temp * G_inv;
                for (int k = 0; k < slm_size; k++)
                {
                    if (Math.Sign(t_vectors_filter_inv[i][k]) == Math.Sign(t_vectors[i][k].Real))
                        correct_estimations_cam_opt += 1.0;
                }
            }
            correct_estimations_cam_opt /= t_vecs_count;

            // Estimate the transmission matrix with a filter for new data
            MathNet.Numerics.LinearAlgebra.Vector<double>[] t_vectors_filter_test_inv =
                new MathNet.Numerics.LinearAlgebra.Vector<double>[t_vecs_count];
            double correct_estimations_test_cam_opt = 0.0;
            for (int i = 0; i < t_vecs_count; i++)
            {
                t_vectors_filter_test_inv[i] = h_results_test_cam[i].Zeta.Real() * G_inv;
                for (int k = 0; k < slm_size; k++)
                {
                    if (Math.Sign(t_vectors_filter_test_inv[i][k]) == Math.Sign(t_vectors_test[i][k].Real))
                        correct_estimations_test_cam_opt += 1.0;
                }
            }
            correct_estimations_test_cam_opt /= t_vecs_count;

            // Save matrices to disk
            //using (BinaryWriter bw = new BinaryWriter(
            //    new FileStream(string.Format("gamma_zeta_{0}_{1}.bin", gamma_zeta_cam.RowCount, gamma_zeta_cam.ColumnCount), FileMode.Create)))
            //{
            //    double[] internal_array = gamma_zeta_cam.AsColumnMajorArray();
            //    for (int i = 0; i < internal_array.Length; i++)
            //        bw.Write(internal_array[i]);
            //}
            //using (BinaryWriter bw = new BinaryWriter(
            //    new FileStream(string.Format("gamma_zeta_inverse_{0}_{1}.bin", gamma_zeta_cam.RowCount, gamma_zeta_cam.ColumnCount), FileMode.Create)))
            //{
            //    double[] internal_array = gamma_zeta_cam.Inverse().AsColumnMajorArray();
            //    for (int i = 0; i < internal_array.Length; i++)
            //        bw.Write(internal_array[i]);
            //}
            //using (BinaryWriter bw = new BinaryWriter(
            //    new FileStream(string.Format("gamma_zeta_tau_{0}_{1}.bin", gamma_zeta_tau_cam.RowCount, gamma_zeta_tau_cam.ColumnCount), FileMode.Create)))
            //{
            //    double[] internal_array = gamma_zeta_tau_cam.AsColumnMajorArray();
            //    for (int i = 0; i < internal_array.Length; i++)
            //        bw.Write(internal_array[i]);
            //}

            Console.WriteLine("■ Ideal case");
            Console.WriteLine("Correct estimations:                        {0}", correct_estimations_ideal);
            Console.WriteLine("■ Camera simulation (with noise)");
            Console.WriteLine("Correct estimations:                        {0}", correct_estimations_cam);
            Console.WriteLine("Correct estimations (opt):                  {0}", correct_estimations_cam_opt);
            Console.WriteLine("Correct estimations (opt, new data):        {0}", correct_estimations_test_cam_opt);
            Console.WriteLine("Average noise:                              {0}", noise_cam_avg);
            Console.WriteLine("Mininal noise:                              {0}", noise_cam_min);
            Console.WriteLine("Maximal noise:                              {0}", noise_cam_max);
            Console.WriteLine("Average noise (true):                       {0}", noise_cam_average);
            Console.WriteLine("Standard deviation (true):                  {0}", noise_cam_stddev);
            Console.WriteLine("Gamma_zeta matrix rank =                    {0}", gamma_zeta_cam.Rank());
            Console.WriteLine("Condition number =                          {0:E2}", gamma_zeta_cam_cond);
            Console.WriteLine("Gamma_zeta eigenvalues =                    {0:E2} ... {1:E2}", gamma_zeta_cam_eig[0],
                 gamma_zeta_cam_eig[gamma_zeta_cam_eig.Count - 1]);
            Console.WriteLine("Gamma_zeta (only noise) eigenvalues =       {0:E2} ... {1:E2}", gamma_zeta_cam_only_noise_eig[0],
                 gamma_zeta_cam_only_noise_eig[gamma_zeta_cam_only_noise_eig.Count - 1]);
            Console.WriteLine("det(gamma_zeta) =                           {0:E2}", gamma_zeta_cam.Determinant());
            Console.WriteLine("Deviation from identity matrix:             {0:E2}", unity_matrix_residual_abs_sum);
            Console.WriteLine("■ Camera simulation (without noise)");
            Console.WriteLine("Correct estimations (cam, noiseless):       {0}", correct_estimations_cam_noiseless);
            Console.WriteLine("Maximal signal:                             {0}", signal_cam_noiseless_max);
            Console.WriteLine();
            Console.WriteLine("Correct estimations (test, ideal):          {0}", correct_estimations_test_ideal);
            Console.WriteLine("Correct estimations (test, cam):            {0}", correct_estimations_test_cam);
            Console.WriteLine("Correct estimations (test, cam, noiseless): {0}", correct_estimations_test_cam_noiseless);

            SaveVector(gamma_zeta_cam_eig, Path.Combine(Path.GetTempPath(), string.Format("{0}.csv", nameof(gamma_zeta_cam_eig))));
            SaveVector(gamma_zeta_cam_only_noise_eig,
                Path.Combine(Path.GetTempPath(), string.Format("{0}.csv", nameof(gamma_zeta_cam_only_noise_eig))));
            SaveMatrix(gamma_zeta_cam, Path.Combine(Path.GetTempPath(), string.Format("{0}.csv", nameof(gamma_zeta_cam))));

            Console.ReadLine();
        }
    }
}
