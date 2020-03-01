using System;

namespace NeuralNetwork.Scripts.NeuronActivatorFunctions
{
    [Serializable]
    public class SigmoidActivator : Activator
    {
        public override double CalculateDerivative(double input)
        {
            return input * (1 - input);
        }

        public override double CalculateValue(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }

        public override ActivatorType GetActivatorType()
        {
            return ActivatorType.Sigmoid;
        }
    }
}