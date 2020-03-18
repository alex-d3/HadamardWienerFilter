using System;
using System.Collections.Generic;
using System.Text;

using MathNet.Numerics.LinearAlgebra;

using ImageSensorSimulator;

namespace HadamardWienerFilter
{
    public static class ImageSensorConstructor
    {
        public static ImageSensor CustomSensor(
            int rows, int cols, double temperature = 323.15, double pga_gain = 1.0, double exposure_time = 0.035,
            bool activate_noise = true)
        {
            ImageSensorProperties properties = new ImageSensorProperties();

            properties.ImageSensorRows = rows;
            properties.ImageSensorColumns = cols;

            properties.PDET_PitchX = 5.0e-6;
            properties.PDET_PitchY = 5.0e-6;

            properties.SIMFL_PhotonShotNoise = activate_noise;
            properties.SIMFL_PhotoelectronsShotNoise = activate_noise;

            properties.SensorKind = SensorKind.CCD;

            properties.Temperature = temperature;

            properties.PDET_FillFactor = 0.95;

            properties.IntegrationTime = exposure_time;
            properties.PDET_ExtQuantumEff = 0.8;
            properties.PDET_QuantumYield = 1.0;

            properties.SIMFL_PhotoResponseNonUniformity = activate_noise;
            properties.PDET_Light_PRNU_Model = NoiseDistribution.JanesickGaussian;
            properties.PDET_Light_PRNU_Sigma = 0.01;

            properties.M_BandGap_0 = 1.166;
            properties.M_VarshniAlpha = 5.5e-04;
            properties.M_VarshniBeta = 636.0;

            properties.SIMFL_DarkCurrent = activate_noise;
            properties.PDET_DarkCurrentFigOfMerit = 1.0 * 1.0e-9 * 1.0e4;
            properties.PDET_DarkCurrentCoef_ca = 4.31e5;
            properties.PDET_DarkCurrentCoef_ed = 2.0;

            properties.SIMFL_DarkCurrentShotNoise = activate_noise;
            // There was a Gaussian noise model in the original code but it was not used.

            properties.SIMFL_DarkCurrentFPN = activate_noise;
            properties.PDET_Dark_PRNU_Model = NoiseDistribution.LogNormal;
            properties.PDET_Dark_FPN_Sigma = 0.4;

            properties.SN_Gain = 5.0e-6;
            properties.SN_V_ref_Reset = 3.1;
            properties.SN_Vs_min = 0.5;
            properties.SIMFL_SenseNodeNonlinearGainResponse = false;// activate_noise;
            properties.SN_GainResponse_k1 = 1.090900000e-14;
            properties.SN_FullEellElectrons = 2.0e4;

            properties.SIMFL_SenseNodeResetNoise = activate_noise;
            properties.SN_ResetNoiseFactor = 0.8;
            // There was a Gaussian reset noise model but it was not used anywhere.

            properties.SF_Gain = 1.0;

            properties.SIMFL_SourceFollowerNonLinearity = activate_noise;
            properties.SF_Nonlinearity_Ratio = 1.05;
            properties.SF_Flicker_Corner_Freq = 1.0e6;
            properties.SF_White_Noise_Density = 15.0e-9;
            properties.SF_Delta_Induced_Modulation = 1.0e-8;
            properties.SF_Data_Clock_Speed = 20.0e6;
            properties.SF_Freq_Sampling_Delta = 10000.0;
            properties.SIMFL_SourceFollowerNoise = activate_noise;

            properties.SIMFL_SourceFollowerFixedPatternOffset = activate_noise;
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
        public static Matrix<double> GenerateTestPattern(ImageSensor sensor, double incident_ph_flux)
        {
            Matrix<double> test_target = Matrix<double>.Build.Dense(sensor.Rows, sensor.Columns);

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

            test_target *= incident_ph_flux;

            test_target = test_target.Transpose();

            return sensor.Simulate(test_target);
        }
    }
}
