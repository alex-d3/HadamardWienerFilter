using System;
using System.Numerics;
using System.IO;
using System.Globalization;
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;

using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using mn_linalg = MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;
using MathNet.Numerics.Distributions;
using CsvHelper;

using ImageSensorSimulator;

namespace HadamardWienerFilter
{
    class Program
    {
        static double SimulateCamera(ImageSensor sensor, double input)
        {
            Matrix<double> sensor_input = Matrix<double>.Build.Dense(sensor.Rows, sensor.Columns);
            double[] sensor_input_arr = sensor_input.AsColumnMajorArray();
            for (int k = 0; k < sensor_input_arr.Length; k++)
                sensor_input_arr[k] = input;
            Matrix<double> sensor_result = sensor.Simulate(sensor_input);
            return sensor_result[0, 0];
        }

        static void Main(string[] args)
        {
            //int slm_size = 256;
            //int t_vecs_count = 1024;
            int slm_size = 64;
            int t_vecs_count = 512;
            // SLM: 64, TVEC: 512
            // SLM: 128, TVEC: 1024
            // SLM: 256, TVEC: 2048
            // SLM: 512, TVEC: 4096
            Complex e_field = new Complex(1.0, 0.0);
            double wavelength = 632.8e-9;
            //Complex e_inc = new Complex(1.0 / Math.Sqrt(2.0), 0.0);
            double e_inc_factor = 1.0 / 4.0;
            Complex e_inc = new Complex(1.0 * Math.Sqrt(e_inc_factor), 0.0);

            int det_rows = 1, det_cols = 1;
            double temperature = 325.15;
            //double pga_gain = 2.0;
            //double integration_time = 0.035;
            double pga_gain = 1.0 * 4.0;
            double integration_time = 1.0 / 25.0;//0.040 / 128.0;

            // Maximum photon shot noise without the influence of dark current
            // e_inc_factor = 8.0 or 9.0
            // pga_gain = 64.0
            // integration_time = 1.0 / 8000.0

            // Maximum dark current
            // e_inc_factor = 1.0 / 4.0
            // pga_gain = 4.0
            // integration_time = 1.0 / 25.0

            double planck_const = 6.62607004e-34;
            double light_speed = 299792458.0;
            double ph_energy = planck_const * light_speed / wavelength;
            double poynting_vec = e_inc.MagnitudeSquared() / (2.0 * 377);
            double ph_flux = poynting_vec / ph_energy;

            double q_eff = 0.8;
            double det_area = 5.0e-6 * 5.0e-6;
            double sensor_electrons = ph_flux * q_eff * det_area * integration_time;

            double well_capacity = 20.0e3;
            double intensity_dn = sensor_electrons / well_capacity * 255.0;


            Console.WriteLine("Photon flux [1/(m^2 s)]: {0:E4}", ph_flux);
            Console.WriteLine("Generated electrons:     {0}", Math.Round(sensor_electrons, 0));
            Console.WriteLine("Intensity [DN]:          {0}", Math.Round(intensity_dn, 0));

            // Create functions that calculate intensity directly and simulate an image sensor
            Func<double, double> ideal_detector = new Func<double, double>((input_intensity) => {
                return input_intensity;
                //double poynting_vec_local = input_intensity / (2.0 * 377.0);
                //double ph_flux_local = poynting_vec_local / ph_energy;
                //double sensor_electrons_local = ph_flux_local * q_eff * det_area * integration_time;
                //double intensity_dn_local = sensor_electrons_local / well_capacity * 255.0;
                //return intensity_dn_local;
            });
            Func<double, double> experimental_detector = new Func<double, double>((input_intensity) => {
                //double poynting_vec_local = input_intensity / (2.0 * 377.0);
                //double ph_flux_local = poynting_vec_local / ph_energy;
                //double sensor_eklectrons = ph_flux_local * q_eff * det_area * integration_time;
                //if (sensor_eklectrons == 0.0)
                //    return 0.0;
                //else if (sensor_eklectrons < 1000.0)
                //    return Poisson.Sample(sensor_eklectrons);
                //else
                //    return Normal.Sample(sensor_eklectrons, Math.Sqrt(sensor_eklectrons));

                double signal = Normal.Sample(input_intensity, 1.0);
                if (signal < 0.0)
                    return 0.0;
                else
                    return signal;

                //return ContinuousUniform.Sample(0.1, 1.0);
            });
            ImageSensor sensor_w_noise = ImageSensorConstructor.CustomSensor(
                det_rows, det_cols, temperature, pga_gain, integration_time, true);
            Func<double, double> advanced_detector = new Func<double, double>((input_intensity) => {
                double poynting_vec_local = input_intensity / (2.0 * 377.0);
                double ph_flux_local = poynting_vec_local / ph_energy;
                return SimulateCamera(sensor_w_noise, ph_flux_local);
            });
            ImageSensor sensor_wo_noise = ImageSensorConstructor.CustomSensor(
                det_rows, det_cols, temperature, pga_gain, integration_time, false);
            Func<double, double> advanced_detector_noiseless = new Func<double, double>((input_intensity) => {
                double poynting_vec_local = input_intensity / (2.0 * 377.0);
                double ph_flux_local = poynting_vec_local / ph_energy;
                return SimulateCamera(sensor_wo_noise, ph_flux_local);
            });

            //double conversion_coefficient = SimulateCamera(sensor_wo_noise, ph_flux);
            double conversion_coefficient = advanced_detector_noiseless(e_inc.MagnitudeSquared());
            Console.WriteLine("Conversion coefficient: {0}", conversion_coefficient);
            Console.WriteLine("w/o noise: {0}", advanced_detector_noiseless(e_inc.MagnitudeSquared()));
            Console.WriteLine("w/  noise: {0}", advanced_detector(e_inc.MagnitudeSquared()));
            //for (int i = 0; i < 100; i++)
            //    Console.WriteLine(advanced_detector(e_inc.MagnitudeSquared()));
            ImageSensor sensor_w_noise_tp = ImageSensorConstructor.CustomSensor(
                512, 512, temperature, pga_gain, integration_time, true);
            ImageSensor sensor_wo_noise_tp = ImageSensorConstructor.CustomSensor(
                512, 512, temperature, pga_gain, integration_time, false);
            ImageSensorConstructor.GenerateTestPattern(
                sensor_w_noise_tp, 2.0 * e_inc.MagnitudeSquared() / (2.0 * 377.0) / ph_energy)
                .WriteToBinary("d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/sensor_w_noise_tp.raw");
            ImageSensorConstructor.GenerateTestPattern(
                sensor_wo_noise_tp, 2.0 * e_inc.MagnitudeSquared() / (2.0 * 377.0) / ph_energy)
                .WriteToBinary("d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/sensor_wo_noise_tp.raw");

            // Create a set of transmission matrices for Wiener filter calculation
            Console.Write("Preparing transmission vectors... ");
            MathNet.Numerics.LinearAlgebra.Vector<Complex>[] t_vectors =
                TransmissionMatrix.GenerateTransmissionVectorsArray(slm_size, t_vecs_count, true);

            //List<mn_linalg::Vector<Complex>> t_vectors_filtered = new List<mn_linalg.Vector<Complex>>(t_vectors.Length);
            //for (int i = 0; i < t_vectors.Length; i++)
            //{
            //    if (experimental_detector((t_vectors[i].Sum() * e_inc).MagnitudeSquared()) != 0.0)
            //    {
            //        t_vectors_filtered.Add(t_vectors[i]);
            //    }
            //}
            //t_vectors = t_vectors_filtered.ToArray();

            //for (int i = 0; i < t_vectors.Length; i++)
            //    Console.WriteLine((t_vectors[i] * t_vectors[i].Conjugate()).MagnitudeSquared());
            Console.WriteLine("DONE");

            Console.Write("Preparing transmission vectors for filter verification... ");
            MathNet.Numerics.LinearAlgebra.Vector<Complex>[] t_vectors_fverif =
                TransmissionMatrix.GenerateTransmissionVectorsArray(slm_size, t_vecs_count, true);
            Console.WriteLine("DONE");

            //mn_linalg::Vector<Complex>[] t_vectors_all =
            //    TransmissionMatrix.GenerateTransmissionVectorsArray(slm_size, 2 * t_vecs_count, true);
            //t_vectors = t_vectors_all.Take(t_vecs_count).ToArray();
            //t_vectors_fverif = t_vectors_all.Skip(t_vecs_count).Take(t_vecs_count).ToArray();


            //List<mn_linalg::Vector<Complex>> t_vectors_fverif_filtered = new List<mn_linalg.Vector<Complex>>(t_vectors_fverif.Length);
            //for (int i = 0; i < t_vectors_fverif.Length; i++)
            //{
            //    if (experimental_detector((t_vectors_fverif[i].Sum() * e_inc).MagnitudeSquared()) != 0.0)
            //    {
            //        t_vectors_fverif_filtered.Add(t_vectors_fverif[i]);
            //    }
            //}
            //t_vectors_fverif = t_vectors_fverif_filtered.ToArray();

            Analyzer a1 = new Analyzer(experimental_detector);
            a1.Analyze(t_vectors, t_vectors_fverif, e_inc, true);
            a1.PrintResults();

            a1.Filter.GammaZeta.WriteToBinary("d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/gamma_zeta.raw");
            a1.Filter.GammaZeta.Svd().W.Diagonal().WriteToCSV("d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/gamma_zeta_sing_vals.csv");
            a1.Filter.GammaZetaEig.WriteToCSV("d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/gamma_zeta_eig.csv");

            string line = "";
            int num = 0;
            while (line != "q")
            {
                Console.Write("Number of a vector to save or q tu quit: ");
                line = Console.ReadLine();
                if (!int.TryParse(line, out num))
                    continue;
                t_vectors[num].WriteToCSV("d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/t_vector_original.csv");
                a1.TransmissionVectorsFiltered[num].WriteToCSV(
                    "d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/t_vector_filtered.csv");
                a1.HResults[num].TransmissionVectorEstimated.WriteToCSV(
                    "d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/t_vector_estimated.csv");
            }
            
            return;
            Console.Write("Simulating focusing... ");
            HadamardResult[] hr_ideal_detector = HadamardAlgorithm.BatchSimulate(ideal_detector, t_vectors, e_inc);
            HadamardResult[] hr_sensor_w_noise = HadamardAlgorithm.BatchSimulate(advanced_detector, t_vectors, e_inc, true);
            HadamardResult[] hr_sensor_w_noise_fverif = HadamardAlgorithm.BatchSimulate(advanced_detector, t_vectors_fverif, e_inc, true);
            HadamardResult[] hr_sensor_wo_noise = HadamardAlgorithm.BatchSimulate(advanced_detector_noiseless, t_vectors, e_inc, true);
            Console.WriteLine("DONE");

            Console.Write("Scaling output... ");
            for (int i = 0; i < hr_sensor_wo_noise.Length; i++)
            {
                hr_ideal_detector[i].TransmissionVectorEstimated /= Math.Sqrt(e_inc_factor);
                hr_ideal_detector[i].Zeta /= Math.Sqrt(e_inc_factor);
                hr_sensor_wo_noise[i].TransmissionVectorEstimated /= Math.Sqrt(conversion_coefficient);
                hr_sensor_wo_noise[i].Zeta /= Math.Sqrt(conversion_coefficient);
                hr_sensor_w_noise[i].TransmissionVectorEstimated /= Math.Sqrt(conversion_coefficient);
                hr_sensor_w_noise[i].Zeta /= Math.Sqrt(conversion_coefficient);

                hr_sensor_w_noise_fverif[i].TransmissionVectorEstimated /= Math.Sqrt(conversion_coefficient);
                hr_sensor_w_noise_fverif[i].Zeta /= Math.Sqrt(conversion_coefficient);
            }
            Console.WriteLine("DONE");

            // Build Wiener filters
            Console.Write("Building Wiener filters... ");
            WienerFilter wf_ideal_detector = new WienerFilter(hr_ideal_detector, t_vectors);
            WienerFilter wf_sensor_w_noise = new WienerFilter(hr_sensor_w_noise, t_vectors);
            WienerFilter wf_sensor_wo_noise = new WienerFilter(hr_sensor_wo_noise, t_vectors);
            Console.WriteLine("DONE");

            Console.Write("Filtering input data... ");
            MathNet.Numerics.LinearAlgebra.Vector<double>[] t_vectors_est_ideal_detector =
                wf_ideal_detector.BatchApply(hr_ideal_detector.Select(e => e.Zeta.Real()).ToArray());
            //MathNet.Numerics.LinearAlgebra.Vector<double>[] t_vectors_est_sensor_w_noise =
            //    wf_sensor_w_noise.BatchApply(hr_ideal_detector.Select(e => e.Zeta.Real()).ToArray());
            //MathNet.Numerics.LinearAlgebra.Vector<double>[] t_vectors_est_sensor_wo_noise =
            //    wf_sensor_wo_noise.BatchApply(hr_ideal_detector.Select(e => e.Zeta.Real()).ToArray());
            MathNet.Numerics.LinearAlgebra.Vector<double>[] t_vectors_est_sensor_w_noise =
                wf_sensor_w_noise.BatchApply(hr_sensor_w_noise.Select(e => e.Zeta.Real()).ToArray());
            MathNet.Numerics.LinearAlgebra.Vector<double>[] t_vectors_est_sensor_wo_noise =
                wf_sensor_wo_noise.BatchApply(hr_sensor_wo_noise.Select(e => e.Zeta.Real()).ToArray());

            MathNet.Numerics.LinearAlgebra.Vector<double>[] t_vectors_est_sensor_w_noise_fverif =
                wf_sensor_w_noise.BatchApply(hr_sensor_w_noise_fverif.Select(e => e.Zeta.Real()).ToArray());
            Console.WriteLine("DONE");

            Console.Write("Calculating correct estimation statistics... ");
            double hr_ideal_detector_cs =
                hr_ideal_detector.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));
            double hr_sensor_w_noise_cs =
                hr_sensor_w_noise.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));
            double hr_sensor_wo_noise_cs =
                hr_sensor_wo_noise.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));

            double hr_sensor_w_noise_cs_fverif =
                hr_sensor_w_noise_fverif.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));

            double hr_ideal_detector_cs_f = 0.0;
            double hr_sensor_w_noise_cs_f = 0.0;
            double hr_sensor_wo_noise_cs_f = 0.0;

            double hr_sensor_w_noise_cs_ffverif = 0.0;
            for (int i = 0; i < t_vectors.Length; i++)
            {
                for (int j = 0; j < t_vectors[j].Count; j++)
                {
                    if (Math.Sign(t_vectors_est_sensor_w_noise[i][j]) == Math.Sign(t_vectors[i][j].Real))
                        hr_sensor_w_noise_cs_f += 1.0;
                    if (Math.Sign(t_vectors_est_sensor_wo_noise[i][j]) == Math.Sign(t_vectors[i][j].Real))
                        hr_sensor_wo_noise_cs_f += 1.0;
                    if (Math.Sign(t_vectors_est_ideal_detector[i][j]) == Math.Sign(t_vectors[i][j].Real))
                        hr_ideal_detector_cs_f += 1.0;

                    if (Math.Sign(t_vectors_est_sensor_w_noise_fverif[i][j]) == Math.Sign(t_vectors_fverif[i][j].Real))
                        hr_sensor_w_noise_cs_ffverif += 1.0;
                }
            }
            hr_ideal_detector_cs_f /= t_vectors.Length;
            hr_sensor_w_noise_cs_f /= t_vectors.Length;
            hr_sensor_wo_noise_cs_f /= t_vectors.Length;
            hr_sensor_w_noise_cs_ffverif /= t_vectors_fverif.Length;


            // Count wrong to right and right to wrong estimations
            double hr_ideal_detector_cs_rtw = 0.0;
            double hr_ideal_detector_cs_wtr = 0.0;
            double hr_sensor_w_noise_cs_rtw = 0.0;
            double hr_sensor_w_noise_cs_wtr = 0.0;
            double hr_sensor_wo_noise_cs_rtw = 0.0;
            double hr_sensor_wo_noise_cs_wtr = 0.0;

            double hr_sensor_w_noise_cs_fverif_rtw = 0.0;
            double hr_sensor_w_noise_cs_fverif_wtr = 0.0;
            for (int i = 0; i < t_vectors.Length; i++)
            {
                for (int j = 0; j < t_vectors[j].Count; j++)
                {
                    
                    if (Math.Sign(hr_ideal_detector[i].TransmissionVectorEstimated[j].Real) != Math.Sign(t_vectors[i][j].Real) &&
                        Math.Sign(t_vectors_est_ideal_detector[i][j]) == Math.Sign(t_vectors[i][j].Real))
                        hr_ideal_detector_cs_wtr += 1.0;
                    else if (Math.Sign(hr_ideal_detector[i].TransmissionVectorEstimated[j].Real) == Math.Sign(t_vectors[i][j].Real) &&
                        Math.Sign(t_vectors_est_ideal_detector[i][j]) != Math.Sign(t_vectors[i][j].Real))
                        hr_ideal_detector_cs_rtw += 1.0;

                    if (Math.Sign(hr_sensor_wo_noise[i].TransmissionVectorEstimated[j].Real) != Math.Sign(t_vectors[i][j].Real) &&
                        Math.Sign(t_vectors_est_sensor_wo_noise[i][j]) == Math.Sign(t_vectors[i][j].Real))
                        hr_sensor_wo_noise_cs_wtr += 1.0;
                    else if (Math.Sign(hr_sensor_wo_noise[i].TransmissionVectorEstimated[j].Real) == Math.Sign(t_vectors[i][j].Real) &&
                        Math.Sign(t_vectors_est_sensor_wo_noise[i][j]) != Math.Sign(t_vectors[i][j].Real))
                        hr_sensor_wo_noise_cs_rtw += 1.0;

                    if (Math.Sign(hr_sensor_w_noise[i].TransmissionVectorEstimated[j].Real) != Math.Sign(t_vectors[i][j].Real) &&
                        Math.Sign(t_vectors_est_sensor_w_noise[i][j]) == Math.Sign(t_vectors[i][j].Real))
                        hr_sensor_w_noise_cs_wtr += 1.0;
                    else if (Math.Sign(hr_sensor_w_noise[i].TransmissionVectorEstimated[j].Real) == Math.Sign(t_vectors[i][j].Real) &&
                        Math.Sign(t_vectors_est_sensor_w_noise[i][j]) != Math.Sign(t_vectors[i][j].Real))
                        hr_sensor_w_noise_cs_rtw += 1.0;


                    if (Math.Sign(hr_sensor_w_noise_fverif[i].TransmissionVectorEstimated[j].Real) != Math.Sign(t_vectors_fverif[i][j].Real) &&
                        Math.Sign(t_vectors_est_sensor_w_noise_fverif[i][j]) == Math.Sign(t_vectors_fverif[i][j].Real))
                        hr_sensor_w_noise_cs_fverif_wtr += 1.0;
                    else if (Math.Sign(hr_sensor_w_noise_fverif[i].TransmissionVectorEstimated[j].Real) == Math.Sign(t_vectors_fverif[i][j].Real) &&
                        Math.Sign(t_vectors_est_sensor_w_noise_fverif[i][j]) != Math.Sign(t_vectors_fverif[i][j].Real))
                        hr_sensor_w_noise_cs_fverif_rtw += 1.0;
                }
            }

            MathNet.Numerics.LinearAlgebra.Vector<Complex>[] slm_sensor_w_noise = t_vectors_est_sensor_w_noise.Select(e => {
                MathNet.Numerics.LinearAlgebra.Vector<Complex> res =
                    MathNet.Numerics.LinearAlgebra.Vector<Complex>.Build.Dense(e.Count);
                for (int i = 0; i < e.Count; i++)
                    if (e[i] >= 0.0)
                        res[i] = 1.0;
                return res;
            }).ToArray();
            MathNet.Numerics.LinearAlgebra.Vector<Complex>[] slm_sensor_wo_noise = t_vectors_est_sensor_wo_noise.Select(e => {
                MathNet.Numerics.LinearAlgebra.Vector<Complex> res =
                    MathNet.Numerics.LinearAlgebra.Vector<Complex>.Build.Dense(e.Count);
                for (int i = 0; i < e.Count; i++)
                    if (e[i] >= 0.0)
                        res[i] = 1.0;
                return res;
            }).ToArray();

            MathNet.Numerics.LinearAlgebra.Vector<Complex>[] slm_sensor_w_noise_fverif = t_vectors_est_sensor_w_noise_fverif.Select(e => {
                MathNet.Numerics.LinearAlgebra.Vector<Complex> res =
                    MathNet.Numerics.LinearAlgebra.Vector<Complex>.Build.Dense(e.Count);
                for (int i = 0; i < e.Count; i++)
                    if (e[i] >= 0.0)
                        res[i] = 1.0;
                return res;
            }).ToArray();

            double ideal_detector_opt_int = 0.0;
            double sensor_w_noise_opt_int = 0.0;
            double sensor_w_noise_f_opt_int = 0.0;
            double sensor_wo_noise_opt_int = 0.0;
            double sensor_wo_noise_f_opt_int = 0.0;

            double sensor_w_noise_opt_int_fverif = 0.0;
            double sensor_w_noise_f_opt_int_fverif = 0.0;
            for (int i = 0; i < t_vectors.Length; i++)
            {
                ideal_detector_opt_int += ideal_detector(((t_vectors[i] * e_inc) * hr_ideal_detector[i].SLMPatternOptimized).MagnitudeSquared());
                sensor_w_noise_opt_int += ideal_detector(((t_vectors[i] * e_inc) * hr_sensor_w_noise[i].SLMPatternOptimized).MagnitudeSquared());
                sensor_w_noise_f_opt_int += ideal_detector(((t_vectors[i] * e_inc) * slm_sensor_w_noise[i]).MagnitudeSquared());
                sensor_wo_noise_f_opt_int += ideal_detector(((t_vectors[i] * e_inc) * slm_sensor_wo_noise[i]).MagnitudeSquared());
                sensor_wo_noise_opt_int += ideal_detector(((t_vectors[i] * e_inc) * hr_sensor_wo_noise[i].SLMPatternOptimized).MagnitudeSquared());

                sensor_w_noise_opt_int_fverif += ideal_detector(((t_vectors_fverif[i] * e_inc) * hr_sensor_w_noise_fverif[i].SLMPatternOptimized).MagnitudeSquared());
                sensor_w_noise_f_opt_int_fverif += ideal_detector(((t_vectors_fverif[i] * e_inc) * slm_sensor_w_noise_fverif[i]).MagnitudeSquared());
            }
            ideal_detector_opt_int /= t_vecs_count;
            sensor_w_noise_opt_int /= t_vecs_count;
            sensor_w_noise_f_opt_int /= t_vecs_count;
            sensor_wo_noise_f_opt_int /= t_vecs_count;
            sensor_wo_noise_opt_int /= t_vecs_count;

            sensor_w_noise_opt_int_fverif /= t_vecs_count;
            sensor_w_noise_f_opt_int_fverif /= t_vecs_count;

            double sensor_w_noise_avg_int =
                hr_sensor_w_noise.Average(new Func<HadamardResult, double>(e => (e.IntensityPlus.Average() + e.IntensityMinus.Average()) / 2.0));
            double sensor_w_noise_max_int =
                hr_sensor_w_noise.Average(new Func<HadamardResult, double>(e => Math.Max(e.IntensityPlus.Max(), e.IntensityMinus.Max())));
            double sensor_w_noise_min_int =
                hr_sensor_w_noise.Average(new Func<HadamardResult, double>(e => Math.Max(e.IntensityPlus.Min(), e.IntensityMinus.Min())));

            double sensor_wo_noise_avg_int =
                hr_sensor_wo_noise.Average(new Func<HadamardResult, double>(e => (e.IntensityPlus.Average() + e.IntensityMinus.Average()) / 2.0));
            double sensor_wo_noise_max_int =
                hr_sensor_wo_noise.Average(new Func<HadamardResult, double>(e => Math.Max(e.IntensityPlus.Max(), e.IntensityMinus.Max())));
            double sensor_wo_noise_min_int =
                hr_sensor_wo_noise.Average(new Func<HadamardResult, double>(e => Math.Max(e.IntensityPlus.Min(), e.IntensityMinus.Min())));


            double sensor_w_noise_avg_int_fverif =
                hr_sensor_w_noise_fverif.Average(new Func<HadamardResult, double>(e => (e.IntensityPlus.Average() + e.IntensityMinus.Average()) / 2.0));
            double sensor_w_noise_max_int_fverif =
                hr_sensor_w_noise_fverif.Average(new Func<HadamardResult, double>(e => Math.Max(e.IntensityPlus.Max(), e.IntensityMinus.Max())));
            double sensor_w_noise_min_int_fverif =
                hr_sensor_w_noise_fverif.Average(new Func<HadamardResult, double>(e => Math.Max(e.IntensityPlus.Min(), e.IntensityMinus.Min())));

            Console.WriteLine("DONE");

            Console.WriteLine("- Ideal detector");
            Console.WriteLine("WtR - RtW switches:                   {0}", hr_ideal_detector_cs_wtr - hr_ideal_detector_cs_rtw);
            Console.WriteLine("Optimized intensity:                  {0:E4}", ideal_detector_opt_int);
            //Console.WriteLine("Optimized intensity (filtered): {0:E4}", ideal_detector_opt_int);
            Console.WriteLine("Correct estimations:                  {0}", hr_ideal_detector_cs);
            Console.WriteLine("- Sensor w/o noise");
            Console.WriteLine("Correct estimations:                  {0}", hr_sensor_wo_noise_cs);
            Console.WriteLine("Correct estimations (filtered):       {0}", hr_sensor_wo_noise_cs_f);
            Console.WriteLine("WtR - RtW switches:                   {0}", hr_sensor_wo_noise_cs_wtr - hr_sensor_wo_noise_cs_rtw);
            Console.WriteLine("Optimized intensity (true):           {0:E4}", sensor_wo_noise_opt_int);
            Console.WriteLine("Optimized intensity (true, filtered): {0:E4}", sensor_wo_noise_f_opt_int);
            Console.WriteLine("Average intensity (camera):           {0}", Math.Round(sensor_wo_noise_avg_int, 0));
            Console.WriteLine("Minimal intensity (camera):           {0}", Math.Round(sensor_wo_noise_min_int, 0));
            Console.WriteLine("Maximal intensity (camera):           {0}", Math.Round(sensor_wo_noise_max_int, 0));
            Console.WriteLine("- Sensor w/ noise");
            Console.WriteLine("Correct estimations:                  {0}", hr_sensor_w_noise_cs);
            Console.WriteLine("Correct estimations (filtered):       {0}", hr_sensor_w_noise_cs_f);
            Console.WriteLine("WtR - RtW switches:                   {0}", hr_sensor_w_noise_cs_wtr - hr_sensor_w_noise_cs_rtw);
            Console.WriteLine("Optimized intensity (true):           {0:E4}", sensor_w_noise_opt_int);
            Console.WriteLine("Optimized intensity (true, filtered): {0:E4}", sensor_w_noise_f_opt_int);
            Console.WriteLine("Average intensity (camera):           {0}", Math.Round(sensor_w_noise_avg_int, 0));
            Console.WriteLine("Minimal intensity (camera):           {0}", Math.Round(sensor_w_noise_min_int, 0));
            Console.WriteLine("Maximal intensity (camera):           {0}", Math.Round(sensor_w_noise_max_int, 0));

            Console.WriteLine("- Sensor w/ noise (verification)");
            Console.WriteLine("Correct estimations:                  {0}", hr_sensor_w_noise_cs_fverif);
            Console.WriteLine("Correct estimations (filtered):       {0}", hr_sensor_w_noise_cs_ffverif);
            Console.WriteLine("WtR - RtW switches:                   {0}", hr_sensor_w_noise_cs_fverif_wtr - hr_sensor_w_noise_cs_fverif_rtw);
            Console.WriteLine("Optimized intensity (true):           {0:E4}", sensor_w_noise_opt_int_fverif);
            Console.WriteLine("Optimized intensity (true, filtered): {0:E4}", sensor_w_noise_f_opt_int_fverif);
            Console.WriteLine("Average intensity (camera):           {0}", Math.Round(sensor_w_noise_avg_int_fverif, 0));
            Console.WriteLine("Minimal intensity (camera):           {0}", Math.Round(sensor_w_noise_min_int_fverif, 0));
            Console.WriteLine("Maximal intensity (camera):           {0}", Math.Round(sensor_w_noise_max_int_fverif, 0));

            hr_ideal_detector[0].TransmissionVectorEstimated.Real().WriteToCSV(
                "d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/ideal_detector.csv");
            hr_sensor_w_noise[0].TransmissionVectorEstimated.Real().WriteToCSV(
                "d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/sensor_w_noise.csv");
            t_vectors_est_sensor_w_noise[0].WriteToCSV(
                "d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/sensor_w_noise_filtered.csv");
            hr_sensor_wo_noise[0].TransmissionVectorEstimated.Real().WriteToCSV(
                "d:/Wavefront shaping/Tasks/01_CCD_noise_Wiener_filter/sensor_wo_noise.csv");
        }
    }
}
