using System;
using System.Collections.Generic;

namespace NeuralNetwork.Utils
{
    public static class NeuralMathCompute
    {
        public static float Sigmoid(double value) {
            return 1.0f / (1.0f + (float) Math.Exp(-value));
        }

        public static int Linear(double value)
        {
            return value > 0 ? 1 : -1;
        }

        public static int Boolean(double value)
        {
            return value > 0 ? 1 : 0;
        }
        public static double Softmax(double input)
        {
            double sum = LogSumExp(input);
            return Math.Exp(input - sum);
        }
        public static double LogSum(double lnx, double lny)
        {
            if (lnx == Double.NegativeInfinity)
                return lny;
            if (lny == Double.NegativeInfinity)
                return lnx;

            if (lnx > lny)
                return lnx + Log1p(Math.Exp(lny - lnx));

            return lny + Log1p(Math.Exp(lnx - lny));
        }
        public static double LogSumExp(this double array)
        {
            double sum = Double.NegativeInfinity;
            sum = LogSum(array, sum);
           
            return sum;
        }
        public static double Log1p(double x)
        {
            if (x <= -1.0)
                return Double.NaN;

            if (Math.Abs(x) > 1e-4)
                return Math.Log(1.0 + x);

            // Use Taylor approx. log(1 + x) = x - x^2/2 with error roughly x^3/3
            // Since |x| < 10^-4, |x|^3 < 10^-12, relative error less than 10^-8
            return (-0.5 * x + 1.0) * x;
        }

        public static double AverageFromList(List<double> entry)
        {
            double result = 0;
            for (int i = 0; i < entry.Count; i++)
            {
                result += entry[i];
            }

            return result / entry.Count;
        }


    }
}