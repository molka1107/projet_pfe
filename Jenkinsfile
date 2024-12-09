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
        
        stage('Docker Build') {
            steps {
                echo 'Building Docker image...'
                withCredentials([usernamePassword(credentialsId: "${DOCKER_HUB_CREDENTIALS}", passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
                    sh 'echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin'
                    sh 'docker build -t ${DOCKER_HUB_REPO}:latest .'

                }
            }
        }

        stage('Docker Run') {
            steps {
                echo 'Running Docker container...'
                sh 'docker run -d --name mon-app-streamlit -p 8501:8501 ${DOCKER_HUB_REPO}:latest'

            }
        }

        stage('Push Docker Image to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: "${DOCKER_HUB_CREDENTIALS}", passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
                    sh 'echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin'
                    sh 'docker push ${DOCKER_HUB_REPO}:latest'
                }
            }
        }
        
        stage('Deploy with Docker Compose') {
            steps {
                script {
                    echo "Starting Docker Compose"
                    sh "docker compose -f docker-compose.yml up -d"
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
