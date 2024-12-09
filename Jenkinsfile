pipeline {    

    agent any
    environment {
        DOCKER_HUB_REPO = 'molka11/mon-app'  
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
                    sh 'docker run --rm -v $PWD:/app -w /app python:3.9 bash -c "pip install -r requirements.txt"'
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

     
      
         
     
    


        


      
        stage('Deploy to Nexus') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'NexusToken', usernameVariable: 'NEXUS_USERNAME', passwordVariable: 'NEXUS_PASSWORD')]) {
                    echo "Deploying to Nexus"
                    sh 'mvn deploy -DskipTests -DaltDeploymentRepository=deploymentRepo::default::http://localhost:8081/repository/maven-releases/ -Dusername=$NEXUS_USERNAME -Dpassword=$NEXUS_PASSWORD'
                }
            }
        }

     
        
         stage('SonarQube Analysis') {
            steps {
                script {
                    withSonarQubeEnv(SONARQUBE_SERVER) {
                        // Juste exécuter sonar-scanner sans spécifier le projectKey si déjà dans le fichier properties
                        sh 'sonar-scanner'
                    }
                }
            }
        }


      stage('Start Prometheus') {
            steps {
                script {
                    echo "Starting Prometheus"
                    sh "docker start prometheus"
                }
            }
        }

        stage('Start Grafana') {
            steps {
                script {
                    echo "Starting Grafana"
                    sh "docker start grafana"
                }
            }
        }

    

      
    }

}
