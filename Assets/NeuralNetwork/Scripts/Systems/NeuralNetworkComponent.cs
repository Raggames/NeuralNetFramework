using System.Collections.Generic;
using UnityEngine;

namespace NeuralNetwork
{
    [RequireComponent(typeof(NeuralNet))]
    public class NeuralNetworkComponent : MonoBehaviour
    {
        public NeuralNet NeuralNet;
        public NetInput[] NetInput; //use a net input to connect a dynamic value to his NeuralNetInstance 
        public NetOutput[] NetOutput;

        public NeuralNetworkController Controller;
        public bool inputStreamOn;
        private void Start()
        {
            NeuralNet = GetComponent<NeuralNet>();
            if (NetOutput.Length == 0)
            {
                NetOutput = new NetOutput[NeuralNet.OutputLayerConstruct[0].NeuronCount];
            }
            if (NetInput.Length == 0)
            {
                NetOutput = new NetOutput[NeuralNet.InputLayerConstruct[0].NeuronCount];
            }
        }

        public void Restart()
        {
            Controller.InstanceReset();
        }
        
        void InputToNeuralNet(NeuralNet neuralNet)
        {
            if (NeuralNet.IsTraining || NeuralNet.IsExecuting)
            {
                inputStreamOn = true;
                Debug.Log("InputToNeuralNet");
                for (int i = 0; i < NeuralNet.InputLayer.NeuronsInLayer.Count; i++)
                {
                    NeuralNet.InputLayer.NeuronsInLayer[i].Inputs.Clear();
                    NeuralNet.InputLayer.NeuronsInLayer[i].ReceiveInput(NetInput[i].InputValue);
                }
                
            }
            else
            {
                inputStreamOn = false;
            }
        }

        public void InstanceEnd(List<double> parameters, NeuralNetworkController controller)
        {
            this.NeuralNet.OnInstanceEnd(parameters, controller);
        }

        private void Update()
        {
            InputToNeuralNet(NeuralNet);
        }
    }
}