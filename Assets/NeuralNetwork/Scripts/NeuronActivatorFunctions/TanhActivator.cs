using System;


namespace NeuralNetwork.Scripts.NeuronActivatorFunctions
{
    public class TanhActivator : Activator
    {
        public override double CalculateDerivative(double input)
        {
            return 1 - Math.Pow(Math.Tanh(input), 2);
        }

        public override double CalculateValue(double input)
        {
            return Math.Tanh(input);
        }

        public override ActivatorType GetActivatorType()
        {
            return ActivatorType.Tanh;
        }
    }
}