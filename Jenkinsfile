pipeline {    

    agent any
    environment {
        DOCKER_HUB_REPO = 'molka11/mon-app-streamlit'  
        DOCKER_HUB_CREDENTIALS = 'DockerToken' 
    }
  
    stages {  

        stage('Git') {
            steps {  
                git branch: 'master', url: 'https://github.com/molka1107/projet_pfe.git', credentialsId: 'GitToken'
            }        
        }

        stage('Install Dependencies') {
            steps {
                script {
                    sh '''
                    bash -c "
                    python3.9 -m venv venv
                    source venv/bin/activate
                    pip install -r requirements.txt
                    "
                    '''
                }
            }
        }

        stage('Run Test') {
            steps {
                script {
                    sh '''
                    source venv/bin/activate
                    if [ ! -f yolov7/modele_a.pt ]; then
                        echo "Le fichier modèle yolov7/modele_a.pt est introuvable. Veuillez l'ajouter avant de relancer le pipeline."
                        exit 1
                    fi
                    export PYTHONPATH=$PYTHONPATH:/var/lib/jenkins/workspace/projet\\ pfe
                    pytest test_object_detection.py -p no:warnings --junitxml=results.xml
                    '''
                }
            }
        }




        stage('Docker Build') {
            steps {
                withCredentials([usernamePassword(credentialsId: "${DOCKER_HUB_CREDENTIALS}", passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
                    sh 'docker build -t ${DOCKER_HUB_REPO}:latest .'

                }
            }
        }

        stage('Docker Run') {
            steps {
                sh 'docker run -d --name mon-app-streamlit -p 8501:8501 ${DOCKER_HUB_REPO}:latest'

            }
        }

        stage('Push Docker Image To Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: "${DOCKER_HUB_CREDENTIALS}", passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
                    sh 'docker push ${DOCKER_HUB_REPO}:latest'
                }
            }
        }

        stage('Deploy Application') {
            steps {
                script {
                    sh 'docker compose down && docker compose up -d'
                }
            }
        }

        stage('Deploy With Docker Compose') {
            steps {
                script {
                    sh "docker compose -f docker-compose.yml up -d"
                }
            }
        }

        stage('Start Prometheus') {
                steps {
                    script {
                        sh "docker start prometheus"
                    }
                }
            }

        stage('Start Grafana') {
            steps {
                script {
                    sh "docker start grafana"
                }
            }
        }

       
  

    

      
    }

    post {
        always {

            junit 'results.xml' 

            emailext (
                subject: "Notification de pipeline pour le projet",
                body: """
                    Bonjour,

                    Le pipeline Jenkins a été exécuté avec le statut suivant : ${currentBuild.currentResult}.
                
                    Vous pouvez consulter les détails de la build à l'adresse suivante : ${env.BUILD_URL}.

                    Merci,
                    L'équipe DevOps
                """,
                to: 'molka.zahra@esprit.tn'
            )
        }
}

}
