using System;
using System.Numerics;

namespace NeuralNetwork.Scripts.NeuronActivatorFunctions
{
    public class SoftmaxActivator : Activator
    {
        public double[] CalculateDeriviative(double[] input, double[] error)
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

        public double[] CalculateValue(double[] input)
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

        public override ActivatorType GetActivatorType()
        {
            return ActivatorType.Softmax;
        }
    }
}