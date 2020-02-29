using System;
using System.Collections.Generic;
using System.Linq;

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
        public static double[] Softmax(double[] input)
        {
            double[] ar_Exp = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                ar_Exp[i] = Math.Exp(input[i]);
            }
            var sum = ar_Exp.Sum();   
            double[] softMax =new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                softMax[i] = ar_Exp[i] / sum;
            }
            return softMax;
        }

        public static double Exp(double entry)
        {
            return Math.Exp(entry);
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