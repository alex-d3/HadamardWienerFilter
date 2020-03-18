using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;
using System.Linq;

using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using mn_linalg = MathNet.Numerics.LinearAlgebra;

using ImageSensorSimulator;

namespace HadamardWienerFilter
{
    public class Analyzer
    {
        public Func<double, double> Detector { get; set; }
        public double ConversionCoefficient { get; set; }
        public HadamardResult[] HResults { get; set; }
        public HadamardResult[] HResultsVerification { get; set; }
        public WienerFilter Filter { get; set; }
        public double HResultsCorrectSigns { get; set; }
        public double HResultsVerificationCorrectSigns { get; set; }
        public double HResultsFilteredCorrectSigns { get; set; }
        public double HResultsVerificationFilteredCorrectSigns { get; set; }
        public double HResultsFilteredCorrectSigns_WTR { get; set; }
        public double HResultsFilteredCorrectSigns_RTW { get; set; }
        public double HResultsVerificationFilteredCorrectSigns_WTR { get; set; }
        public double HResultsVerificationFilteredCorrectSigns_RTW { get; set; }
        public mn_linalg::Vector<double>[] TransmissionVectorsFiltered { get; set; }
        public mn_linalg::Vector<double>[] TransmissionVectorsVerificationFiltered { get; set; }
        public mn_linalg::Vector<Complex>[] HResultsFilteredSLMPatterns { get; set; }
        public mn_linalg::Vector<Complex>[] HResultsVerificationFilteredSLMPatterns { get; set; }
        public double HResultsOptimizedIntensity { get; set; }
        public double HResultsFilteredOptimizedIntensity { get; set; }
        public double HResultsVerificationOptimizedIntensity { get; set; }
        public double HResultsVerificationFilteredOptimizedIntensity { get; set; }
        public double HResultsAverageIntensity { get; set; }
        public double HResultsMaximalIntensity { get; set; }
        public double HResultsMinimalIntensity { get; set; }
        public double HResultsVerificationAverageIntensity { get; set; }
        public double HResultsVerificationMaximalIntensity { get; set; }
        public double HResultsVerificationMinimalIntensity { get; set; }
        public double MeanSquareError { get; set; }
        public double MeanSquareErrorFiltered { get; set; }
        public double MeanSquareErrorVerification { get; set; }
        public double MeanSquareErrorFilteredVerification { get; set; }
        public Analyzer(Func<double, double> detector)
        {
            Detector = detector;
        }

        public void Analyze(
            mn_linalg::Vector<Complex>[] t_vectors, mn_linalg::Vector<Complex>[] t_vectors_verif, Complex e_inc, bool avoid_zero_i_1)
        {
            // Calculation of the conversion coefficient for normalization
            for (int i = 0; i < 20; i++)
                ConversionCoefficient += Detector(e_inc.MagnitudeSquared());
            ConversionCoefficient /= 20;

            Console.Write("Simulating focusing... ");
            HResults = HadamardAlgorithm.BatchSimulate(Detector, t_vectors, e_inc, avoid_zero_i_1);
            HResultsVerification = HadamardAlgorithm.BatchSimulate(Detector, t_vectors_verif, e_inc, avoid_zero_i_1);
            Console.WriteLine("DONE");

            Console.Write("Scaling output... ");
            for (int i = 0; i < HResults.Length; i++)
            {
                HResults[i].TransmissionVectorEstimated /= Math.Sqrt(ConversionCoefficient);
                HResults[i].Zeta /= Math.Sqrt(ConversionCoefficient);
            }
            for (int i = 0; i < HResultsVerification.Length; i++)
            {
                HResultsVerification[i].TransmissionVectorEstimated /= Math.Sqrt(ConversionCoefficient);
                HResultsVerification[i].Zeta /= Math.Sqrt(ConversionCoefficient);
            }

            Console.WriteLine("DONE");

            // Subtracting mean value of intensities
            //double average_intensity = HResults.Average(e => e.Zeta.Real().Average());
            //for (int i = 0; i < HResults.Length; i++)
            //{
            //    HResults[i].Zeta -= average_intensity;
            //}
            //double average_intensity_verification = HResultsVerification.Average(e => e.Zeta.Real().Average());
            //for (int i = 0; i < HResultsVerification.Length; i++)
            //{
            //    HResultsVerification[i].Zeta -= average_intensity_verification;
            //}

            // Excluding simulations with zero I_1
            //List<HadamardResult> h_results_excl = new List<HadamardResult>(HResults.Length);
            //List<mn_linalg::Vector<Complex>> t_vectors_excl = new List<mn_linalg.Vector<Complex>>(t_vectors.Length);
            //for (int i = 0; i < HResults.Length; i++)
            //{
            //    if (!HResults[i].ZeroInitialIntensity)
            //    {
            //        h_results_excl.Add(HResults[i]);
            //        t_vectors_excl.Add(t_vectors[i]);
            //    }
            //}
            //HResults = h_results_excl.ToArray();
            //mn_linalg::Vector<Complex>[] t_vectors = t_vectors_excl.ToArray();

            // Build Wiener filters
            Console.Write("Building Wiener filters... ");
            Filter = new WienerFilter(HResults, t_vectors);
            //Filter = new WienerFilter(HResultsVerification, t_vectors_verif);
            Console.WriteLine("DONE");

            Console.Write("Filtering input data... ");
            TransmissionVectorsFiltered =
                Filter.BatchApply(HResults.Select(e => e.Zeta.Real()).ToArray());
            TransmissionVectorsVerificationFiltered =
                Filter.BatchApply(HResultsVerification.Select(e => e.Zeta.Real()).ToArray());
            Console.WriteLine("DONE");

            Console.Write("Calculating correct estimation statistics... ");
            HResultsCorrectSigns =
                HResults.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));
            HResultsVerificationCorrectSigns =
                HResultsVerification.Average(new Func<HadamardResult, double>(e => (double)e.CorrectSignsNumber));

            HResultsFilteredCorrectSigns = 0.0;
            HResultsVerificationFilteredCorrectSigns = 0.0;
            for (int i = 0; i < t_vectors.Length; i++)
            {
                for (int j = 0; j < t_vectors[i].Count; j++)
                {
                    if (Math.Sign(TransmissionVectorsFiltered[i][j]) == Math.Sign(t_vectors[i][j].Real))
                        HResultsFilteredCorrectSigns += 1.0;
                }
            }
            HResultsFilteredCorrectSigns /= t_vectors.Length;
            for (int i = 0; i < t_vectors_verif.Length; i++)
            {
                for (int j = 0; j < t_vectors_verif[i].Count; j++)
                {
                    if (Math.Sign(TransmissionVectorsVerificationFiltered[i][j]) == Math.Sign(t_vectors_verif[i][j].Real))
                        HResultsVerificationFilteredCorrectSigns += 1.0;
                }
            }
            HResultsVerificationFilteredCorrectSigns /= t_vectors_verif.Length;

            // Count wrong to right and right to wrong estimations
            HResultsFilteredCorrectSigns_WTR = 0.0;
            HResultsFilteredCorrectSigns_RTW = 0.0;
            for (int i = 0; i < t_vectors.Length; i++)
            {
                double initial_wtr = HResultsFilteredCorrectSigns_WTR;
                double initial_rtw = HResultsFilteredCorrectSigns_RTW;
                for (int j = 0; j < t_vectors[i].Count; j++)
                {
                    // Was wrong and became correct
                    if (Math.Sign(HResults[i].TransmissionVectorEstimated[j].Real) != Math.Sign(t_vectors[i][j].Real) &&
                        Math.Sign(TransmissionVectorsFiltered[i][j]) == Math.Sign(t_vectors[i][j].Real))
                        HResultsFilteredCorrectSigns_WTR += 1.0;
                    // Was correct and became wrong
                    else if (Math.Sign(HResults[i].TransmissionVectorEstimated[j].Real) == Math.Sign(t_vectors[i][j].Real) &&
                        Math.Sign(TransmissionVectorsFiltered[i][j]) != Math.Sign(t_vectors[i][j].Real))
                        HResultsFilteredCorrectSigns_RTW += 1.0;
                }
                if (HResultsFilteredCorrectSigns_WTR - initial_wtr > HResultsFilteredCorrectSigns_RTW - initial_rtw)
                {
                    Console.WriteLine("Became better: {0}", i);
                }
            }
            HResultsVerificationFilteredCorrectSigns_WTR = 0.0;
            HResultsVerificationFilteredCorrectSigns_RTW = 0.0;
            for (int i = 0; i < t_vectors_verif.Length; i++)
            {
                for (int j = 0; j < t_vectors_verif[i].Count; j++)
                {
                    // Was wrong and became correct
                    if (Math.Sign(HResultsVerification[i].TransmissionVectorEstimated[j].Real) != Math.Sign(t_vectors_verif[i][j].Real) &&
                        Math.Sign(TransmissionVectorsVerificationFiltered[i][j]) == Math.Sign(t_vectors_verif[i][j].Real))
                        HResultsVerificationFilteredCorrectSigns_WTR += 1.0;
                    // Was correct and became wrong
                    else if (Math.Sign(HResultsVerification[i].TransmissionVectorEstimated[j].Real) == Math.Sign(t_vectors_verif[i][j].Real) &&
                        Math.Sign(TransmissionVectorsVerificationFiltered[i][j]) != Math.Sign(t_vectors_verif[i][j].Real))
                        HResultsVerificationFilteredCorrectSigns_RTW += 1.0;
                }
            }

            // Calculate the SLM patterns for the filtered data
            HResultsFilteredSLMPatterns = TransmissionVectorsFiltered.Select(e => {
                mn_linalg::Vector<Complex> res = mn_linalg::Vector<Complex>.Build.Dense(e.Count);
                for (int i = 0; i < e.Count; i++)
                    //if (e[i] >= 0.0)
                    if (e[i] > 0.0)
                        res[i] = 1.0;
                return res;
            }).ToArray();
            HResultsVerificationFilteredSLMPatterns = TransmissionVectorsVerificationFiltered.Select(e => {
                mn_linalg::Vector<Complex> res = mn_linalg::Vector<Complex>.Build.Dense(e.Count);
                for (int i = 0; i < e.Count; i++)
                    //if (e[i] >= 0.0)
                    if (e[i] > 0.0)
                        res[i] = 1.0;
                return res;
            }).ToArray();

            // Calculation of optimized intensity for filtered and not filtered data
            HResultsOptimizedIntensity = 0.0;
            HResultsFilteredOptimizedIntensity = 0.0;
            for (int i = 0; i < t_vectors.Length; i++)
            {
                HResultsOptimizedIntensity += Detector(((t_vectors[i] * e_inc) * HResults[i].SLMPatternOptimized).MagnitudeSquared());
                HResultsFilteredOptimizedIntensity += Detector(((t_vectors[i] * e_inc) * HResultsFilteredSLMPatterns[i]).MagnitudeSquared());
            }
            HResultsOptimizedIntensity /= t_vectors.Length;
            HResultsFilteredOptimizedIntensity /= t_vectors.Length;

            HResultsVerificationOptimizedIntensity = 0.0;
            HResultsVerificationFilteredOptimizedIntensity = 0.0;
            for (int i = 0; i < t_vectors_verif.Length; i++)
            {
                HResultsVerificationOptimizedIntensity += Detector(((t_vectors_verif[i] * e_inc) * HResultsVerification[i].SLMPatternOptimized).MagnitudeSquared());
                HResultsVerificationFilteredOptimizedIntensity += Detector(((t_vectors_verif[i] * e_inc) * HResultsVerificationFilteredSLMPatterns[i]).MagnitudeSquared());
            }
            HResultsVerificationOptimizedIntensity /= t_vectors_verif.Length;
            HResultsVerificationFilteredOptimizedIntensity /= t_vectors_verif.Length;

            HResultsAverageIntensity =
                HResults.Average(new Func<HadamardResult, double>(e => (e.IntensityPlus.Average() + e.IntensityMinus.Average()) / 2.0));
            HResultsMaximalIntensity =
                HResults.Average(new Func<HadamardResult, double>(e => Math.Max(e.IntensityPlus.Max(), e.IntensityMinus.Max())));
            HResultsMinimalIntensity =
                HResults.Average(new Func<HadamardResult, double>(e => Math.Max(e.IntensityPlus.Min(), e.IntensityMinus.Min())));

            HResultsVerificationAverageIntensity =
                HResultsVerification.Average(new Func<HadamardResult, double>(e => (e.IntensityPlus.Average() + e.IntensityMinus.Average()) / 2.0));
            HResultsVerificationMaximalIntensity =
                HResultsVerification.Average(new Func<HadamardResult, double>(e => Math.Max(e.IntensityPlus.Max(), e.IntensityMinus.Max())));
            HResultsVerificationMinimalIntensity =
                HResultsVerification.Average(new Func<HadamardResult, double>(e => Math.Max(e.IntensityPlus.Min(), e.IntensityMinus.Min())));

            // Calculate standard deviation of the estimated and filtered transmission vectors
            MeanSquareError = HResults.Select(
                (e, i) => {
                    return (e.TransmissionVectorEstimated.Real() - t_vectors[i].Real()).PointwisePower(2.0);
                }).ToArray().Average(e => e.Average());
            MeanSquareErrorFiltered = TransmissionVectorsFiltered.Select(
                (e, i) => {
                    return (e - t_vectors[i].Real()).PointwisePower(2.0);
                }).ToArray().Average(e => e.Average());

            MeanSquareErrorVerification = HResultsVerification.Select(
                (e, i) => {
                    return (e.TransmissionVectorEstimated.Real() - t_vectors_verif[i].Real()).PointwisePower(2.0);
                }).ToArray().Average(e => e.Average());
            MeanSquareErrorFilteredVerification = TransmissionVectorsVerificationFiltered.Select(
                (e, i) => {
                    return (e - t_vectors_verif[i].Real()).PointwisePower(2.0);
                }).ToArray().Average(e => e.Average());
            

            Console.WriteLine("DONE");
        }
        public void PrintResults()
        {
            Console.WriteLine("- Construction:");
            Console.WriteLine("Correct estimations:                  {0}", HResultsCorrectSigns);
            Console.WriteLine("Correct estimations (filtered):       {0}", HResultsFilteredCorrectSigns);
            Console.WriteLine("WtR - RtW switches:                   {0}", HResultsFilteredCorrectSigns_WTR -
                HResultsFilteredCorrectSigns_RTW);
            Console.WriteLine("Optimized intensity (true):           {0:0.00}", HResultsOptimizedIntensity);
            Console.WriteLine("Optimized intensity (true, filtered): {0:0.00}", HResultsFilteredOptimizedIntensity);
            Console.WriteLine("Average intensity:                    {0}", HResultsAverageIntensity);
            Console.WriteLine("Minimal intensity:                    {0}", HResultsMinimalIntensity);
            Console.WriteLine("Maximal intensity:                    {0}", HResultsMaximalIntensity);
            Console.WriteLine("Mean square error:                    {0}", MeanSquareError);
            Console.WriteLine("Mean square error (filtered):         {0}", MeanSquareErrorFiltered);

            Console.WriteLine("- Verification:");
            Console.WriteLine("Correct estimations:                  {0}", HResultsVerificationCorrectSigns);
            Console.WriteLine("Correct estimations (filtered):       {0}", HResultsVerificationFilteredCorrectSigns);
            Console.WriteLine("WtR - RtW switches:                   {0}", HResultsVerificationFilteredCorrectSigns_WTR -
                HResultsVerificationFilteredCorrectSigns_RTW);
            Console.WriteLine("Optimized intensity (true):           {0:0.00}", HResultsVerificationOptimizedIntensity);
            Console.WriteLine("Optimized intensity (true, filtered): {0:0.00}", HResultsVerificationFilteredOptimizedIntensity);
            Console.WriteLine("Average intensity:                    {0}", HResultsVerificationAverageIntensity);
            Console.WriteLine("Minimal intensity:                    {0}", HResultsVerificationMinimalIntensity);
            Console.WriteLine("Maximal intensity:                    {0}", HResultsVerificationMaximalIntensity);
            Console.WriteLine("Mean square error:                    {0}", MeanSquareErrorVerification);
            Console.WriteLine("Mean square error (filtered):         {0}", MeanSquareErrorFilteredVerification);
        }
    }
}
