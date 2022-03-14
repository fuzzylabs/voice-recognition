FROM zenmldocker/zenml:0.6.2
RUN mkdir /numba_cache
RUN chmod 777 /numba_cache
ENV NUMBA_CACHE_DIR="/numba_cache"

# Set up libsndfile
# n.b. for this, we need to set the timezone up, or the libsndfile installer will ask for it interactively
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt install -y libsndfile-dev