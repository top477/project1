const reportWebVitals = onPerfEntry => {
  // Check if the provided callback is a function
  if (onPerfEntry && onPerfEntry instanceof Function) {
    // Dynamically import the 'web-vitals' module
    import('web-vitals').then(({ getCLS, getFID, getFCP, getLCP, getTTFB }) => {
      // Call the metrics collection functions with the callback
      getCLS(onPerfEntry); // Collect and report Cumulative Layout Shift (CLS) metrics
      getFID(onPerfEntry); // Collect and report First Input Delay (FID) metrics
      getFCP(onPerfEntry); // Collect and report First Contentful Paint (FCP) metrics
      getLCP(onPerfEntry); // Collect and report Largest Contentful Paint (LCP) metrics
      getTTFB(onPerfEntry); // Collect and report Time to First Byte (TTFB) metrics
    });
  }
};

export default reportWebVitals;
