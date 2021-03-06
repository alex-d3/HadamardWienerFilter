﻿using System;
using System.Collections.Generic;
using System.Text;
using System.Numerics;

using MathNet.Numerics.LinearAlgebra;
using CsvHelper;

namespace HadamardWienerFilter
{
    public struct HadamardResult
    {
        public MathNet.Numerics.LinearAlgebra.Vector<Complex> TransmissionVectorEstimated { get; set; }
        public MathNet.Numerics.LinearAlgebra.Vector<Complex> SLMPatternOptimized { get; set; }
        public MathNet.Numerics.LinearAlgebra.Vector<double> IntensityPlus { get; set; }
        public MathNet.Numerics.LinearAlgebra.Vector<double> IntensityMinus { get; set; }
        public MathNet.Numerics.LinearAlgebra.Vector<double> ReSigma { get; set; }
        public MathNet.Numerics.LinearAlgebra.Vector<Complex> Zeta { get; set; }
        public int CorrectSignsNumber { get; set; }
        public double Enhancement { get; set; }
        public double OptimizedIntensity { get; set; }
        public bool ZeroInitialIntensity { get; set; }
    }
}
