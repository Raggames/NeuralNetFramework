namespace NeuralNetwork.Scripts.Data
{
    public class NeuralNetworkErrorParameter
    {
        public double EvaluationParameter;
        public EParameterType ParameterType;
    }

    public enum EParameterType
    {
        ShouldBeInferior,
        ShouldBeSuperior,
    }
}