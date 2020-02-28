using System.Collections;
using System.Collections.Generic;
using NeuralNetwork.Scripts.Controllers;
using UnityEngine;

public class BulletComponent : MonoBehaviour
{
    public ShooterController controller;
    private void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.CompareTag("Tank"))
        {
            if (controller.combo) controller.Points++;
            controller.Points+=2;
            controller.combo = true;
            controller.FireTimer += 0.7f;
            other.gameObject.GetComponent<ShooterController>().OnInstanceFail();
            Destroy(gameObject);
        }
        if(other.gameObject.CompareTag("Bullet")) 
        {
            //Dunuthing
            Physics.IgnoreCollision(this.GetComponent<Collider>(), other.collider, true);
        }

        if (other.gameObject.CompareTag("Wall"))
        {
          
            Debug.Log("Destroy");
            controller.Points--;
            controller.combo = false;
            Destroy(gameObject);
        }
        
    }
}
