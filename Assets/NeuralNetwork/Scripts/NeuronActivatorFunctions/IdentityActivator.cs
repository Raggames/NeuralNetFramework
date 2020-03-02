namespace NeuralNetwork.Scripts.NeuronActivatorFunctions
{
    public class IdentityActivator : Activator
    {
        public override double CalculateDerivative(double input)
        {
            return input;
        }
        public override double CalculateValue(double input)
        {
            return input;
        }
        public override ActivatorType GetActivatorType()
        {
            return ActivatorType.Identity;
        }
    }
}