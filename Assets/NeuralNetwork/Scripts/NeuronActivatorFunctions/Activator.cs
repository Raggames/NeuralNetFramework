using System;
using System.Runtime.InteropServices.WindowsRuntime;

namespace NeuralNetwork.Scripts.NeuronActivatorFunctions
{
    [Serializable]
    public abstract class Activator
    {
        public virtual double CalculateValue(double input)
        {
           return new double();
        }

        public virtual double CalculateDerivative(double input)
        {
            return new double();
        }

        public virtual double[] CalculateValues(double[] input)
        {
            return new double[0];
        }
        public virtual double[] CalculateDerivatives(double[] input)
        {
            return new double[0];
        }
        public abstract ActivatorType GetActivatorType();
    }

    public enum ActivatorType
    {
        Identity,
        Tanh,
        Relu,
        Sigmoid,
        Softmax,
    }
}