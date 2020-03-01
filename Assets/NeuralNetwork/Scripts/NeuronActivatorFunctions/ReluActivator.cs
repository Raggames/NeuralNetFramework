using System;


namespace NeuralNetwork.Scripts.NeuronActivatorFunctions
{
    [Serializable]
    public class ReluActivator : Activator
    {
        public override double CalculateDerivative(double input)
        {
            return input == 0 ? 0 : 1;
        }

        public override double CalculateValue(double input)
        {
            return Math.Max(0, input);
        }

        public override ActivatorType GetActivatorType()
        {
            return ActivatorType.Relu;
        }
    }
}