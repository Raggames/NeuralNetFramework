using System;
using MathNet.Numerics.LinearAlgebra.Double;

namespace NeuralNetwork.Scripts.NetToolbox
{
    public static class NetActivators{}
    public static class IdentityActivator 
    {
        public static double CalculateDerivative(double input)
        {
            return input;
        }
        public static double CalculateValue(double input)
        {
            return input;
        }
        
    }
    public static class TanhActivator 
    {
        public static double CalculateDerivative(double input)
        {
            return 1 - Math.Pow(Math.Tanh(input), 2);
        }

        public static double CalculateValue(double input)
        {
            return Math.Tanh(input);
        }

       
    }
    public static class ReluActivator 
    {
        public static double CalculateDerivative(double input)
        {
            return input == 0 ? 0 : 1;
        }

        public static double CalculateValue(double input)
        {
            return Math.Max(0, input);
        }

        
    }
    public static class SigmoidActivator 
    {
        public static double CalculateDerivative(double input)
        {
            return input * (1 - input);
        }

        public static double CalculateValue(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }

      
    }
    public static class SoftmaxActivator 
    {
        public static double[] CalculateDerivative(double[] input, double[] error)
        {
            var jacobian = Matrix.Build.Dense(input.Length, input.Length);
            var vector = Vector.Build.Dense(error);

            jacobian.MapIndexedInplace((i, j, q) =>
            {
                var o1 = input[i];
                var o2 = input[j];
                return i == j ? o1 * (1 - o1) : -o1 * o2;
            });

            return jacobian.Multiply(vector).ToArray();
        }

        public static double[] CalculateValue(double[] input)
        {
            var output = new double[input.Length];
            double sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                sum += Math.Exp(input[i]);
            }

            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Math.Exp(input[i]) / sum;
            }

            return output;
        }
      
    }

    public static class AverageActivator
    {
        public static double CalculateDerivative(double input, int prevNeur=1)
        {
            return input * prevNeur;
        }

        public static double CalculateValue(double input, int prevNeur=1)
        {
           return input/prevNeur;
        }
    }
    public enum ActivatorType
    {
        Identity,
        Average,
        Tanh,
        Relu,
        Sigmoid,
        Softmax,
    }
}