using System.IO;
using NeuralNetwork.Scripts.Data;
using UnityEngine;

namespace NeuralNetwork
{
    public static class NeuralNetworkSerializer
    {
        public static void Save(NetData gameData, string fileName)
        {
            string saveJson = JsonUtility.ToJson(gameData);
            File.WriteAllText(Application.dataPath + "/StreamingAssets/" + fileName, saveJson);
            Debug.Log("NetData Saved");
        }
        public static NetData Load(NetData netData, string fileName)
        {
            NetData loadedData = new NetData();
            if (File.Exists(Application.dataPath + "/StreamingAssets/" + fileName))
            {
                string LoadJson = File.ReadAllText(Application.dataPath + "/StreamingAssets/" + fileName);
                loadedData = JsonUtility.FromJson<NetData>(LoadJson);
                Debug.Log("Loaded :" + fileName);
            }
            return loadedData;
        }
        
        public static void GenericSave<T>(T data, string fileName)
        {
            string saveJson = JsonUtility.ToJson(data);
            File.WriteAllText(Application.dataPath + "/StreamingAssets/" + fileName, saveJson);
            Debug.Log("NetData Saved");
        }
        public static T GenericLoad<T>(T data, string fileName)
        {
            if (File.Exists(Application.dataPath + "/StreamingAssets/" + fileName))
            {
                string LoadJson = File.ReadAllText(Application.dataPath + "/StreamingAssets/" + fileName);
                data = JsonUtility.FromJson<T>(LoadJson);
                Debug.Log("Loaded :" + fileName);
            }
            return data;
        }
        
    }
}